"""双大核 + 微专家池的前向实现。

结构分三层（对应 README 的 Tier 1/2/3）：

    Tier 1  big_arts   原版 MLP，完全冻结，语言底座
    Tier 2  big_sci    克隆 MLP，低学习率微调，理科能力
    Tier 3  lora_A/B   32 路 rank-r LoRA 堆叠权重，路由后按需生效

``DualBigLittleMoE`` 是训练与推理共用的基类，持有全部结构参数与
双大核混音；两个子类只覆盖**专家选择策略**：

    TrainMoE   逐位置稠密 top-k（每个 token 各自选专家）
    InferMoE   末 token top-k + pinned 主机内存流式搬运

把公共部分收进基类是为了消灭此前两份 wrapper 各自维护一份超参、
只靠注释人工同步的问题。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .config import Config


class RouterStats:
    """一次前向中路由器的观测值。

    训练时辅助损失需要这些张量。与其把中间量挂在 module 的 ``last_*``
    属性上（两份 wrapper 对同一属性的解读曾不一致），不如显式返回。
    """

    __slots__ = ("big_logits", "little_logits", "little_probs", "topk_weights")

    def __init__(self, big_logits, little_logits, little_probs, topk_weights):
        self.big_logits = big_logits
        self.little_logits = little_logits
        self.little_probs = little_probs
        self.topk_weights = topk_weights


class DualBigLittleMoE(nn.Module):
    """双大核 + 微专家池的公共基类。

    子类必须实现 :meth:`expert_output`，返回微专家对激活的贡献
    （已含 LoRA scaling，但**不含** gamma）。
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        cfg: Config,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.gamma = cfg.gamma
        self.scaling = cfg.scaling
        self.rank = cfg.lora_rank
        self.device = device or cfg.device
        self.dtype = dtype or cfg.torch_dtype

        # Tier 1：原版 MLP，原地保留并彻底冻结
        self.big_arts = original_mlp
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # Tier 2：克隆一份，接受低学习率微调
        self.big_sci = copy.deepcopy(original_mlp).to(self.device)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 路由器（无 bias）
        self.router_big = nn.Linear(
            hidden_dim, 2, bias=False, device=self.device, dtype=self.dtype
        )
        self.router_little = nn.Linear(
            hidden_dim,
            cfg.num_experts,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        # Tier 3：堆叠 LoRA 权重，形状 (E, r, D)
        self.lora_A = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                cfg.lora_rank,
                hidden_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty_like(self.lora_A)
        )
        self.reset_expert_params_()

    # ---- 构造辅助 ----
    def reset_expert_params_(self) -> None:
        """A 用 kaiming，B 置零 —— 保证初始输出恒为 0，不破坏基座行为。"""
        nn.init.kaiming_uniform_(
            self.lora_A.view(-1, self.hidden_dim), a=math.sqrt(5)
        )
        nn.init.zeros_(self.lora_B)

    # ---- 公共前向片段 ----
    def big_core(self, x: torch.Tensor):
        """双大核逐位置混音，返回 ``(输出, 路由概率)``。

        训练与推理共用同一份实现，确保大核部分永不错位。
        """
        logits = self.router_big(x)
        weights = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        out = weights[..., 0:1] * self.big_arts(x) + weights[
            ..., 1:2
        ] * self.big_sci(x)
        return out, weights, logits

    def select_topk(self, logits: torch.Tensor):
        """top-k + 组内归一化，返回 ``(权重, 下标, 稠密权重)``。"""
        probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        dense = torch.zeros_like(probs).scatter_(-1, topi, topv)
        return topv, topi, dense, probs

    # ---- 子类实现 ----
    def expert_output(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out, _, _ = self.big_core(x)
        return big_out + self.gamma * self.expert_output(x)

    # ---- 参数分组 ----
    def parameter_groups(self) -> list[dict]:
        """按学习率量级分组的参数，供优化器使用。"""
        big_sci, routers, experts = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("big_sci."):
                big_sci.append(p)
            elif name.startswith(("router_big.", "router_little.")):
                routers.append(p)
            elif name.startswith(("lora_A", "lora_B")):
                experts.append(p)
        cfg = self.cfg
        wd = cfg.weight_decay
        return [
            {"params": big_sci, "lr": cfg.lr_big_sci, "weight_decay": wd,
             "name": "big_sci"},
            {"params": routers, "lr": cfg.lr_router, "weight_decay": wd,
             "name": "router"},
            {"params": experts, "lr": cfg.lr_experts, "weight_decay": wd,
             "name": "experts"},
        ]


class TrainMoE(DualBigLittleMoE):
    """训练路径：逐位置稠密 top-k。

    稠密算完再按 top-k 权重掩码，有两个好处：
      * 不物化 ``(N, E, D)`` 激活，显存可控
      * 专家权重的梯度路径不被 ModuleList 式索引切断

    注意：``collect_stats`` 为 False 时（默认）``forward`` 只返回 Tensor，
    与 decoder layer 的约定一致；训练循环与评估脚本将其置 True 以取回
    :class:`RouterStats`。
    """

    #: 置 True 时 forward 把 RouterStats 写入 ``self.last_stats``
    collect_stats: bool = False
    last_stats: RouterStats | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterStats]:
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen

        big_out, _, big_logits = self.big_core(x)
        little_logits = self.router_little(x)
        topv, topi, dense, probs = self.select_topk(little_logits)

        a_flat = self.lora_A.reshape(-1, dim)
        b_flat = self.lora_B.reshape(-1, dim)
        h = torch.matmul(x.reshape(n, dim), a_flat.t())          # (N, E*r)
        w_exp = dense.reshape(n, -1, 1).expand(n, -1, self.rank)
        h = h * w_exp.reshape(n, -1)
        lora = torch.matmul(h, b_flat).reshape(bsz, seqlen, dim) * self.scaling

        out = big_out + self.gamma * lora
        if self.collect_stats:
            # 走 side channel 而非返回值：transformers 5.x 会给
            # layer.__call__ 套 wrapper，forward hook 若返回 tuple，
            # 该 tuple 会被当作 hidden_states 传给下一层而报错。
            # 调用方在 forward 之后立即读取本属性。
            self.last_stats = RouterStats(
                big_logits, little_logits, probs, dense
            )
        return out


class InferMoE(DualBigLittleMoE):
    """推理路径：末 token 选专家 + pinned 主机内存流式搬运。

    与训练路径的**唯一**差异是专家选择时机：这里用最后一个 token 决定
    整段 prompt 的专家集合，否则每多一个 token 就要多搬一轮专家。
    decode 阶段（seqlen=1）两者等价。
    """

    def __init__(self, *args, num_staging_buffers: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_staging_buffers = num_staging_buffers
        self.host_expert_pool: dict[str, torch.Tensor] = {}
        self.telemetry = self._new_telemetry()

    def _new_telemetry(self) -> dict[str, torch.Tensor]:
        return {
            "arts_weight": torch.zeros((), device=self.device, dtype=torch.float32),
            "sci_weight": torch.zeros((), device=self.device, dtype=torch.float32),
            "expert_calls": torch.zeros(
                self.num_experts, device=self.device, dtype=torch.long
            ),
        }

    def load_expert_pool(self, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        """把训练好的堆叠权重拷到 pinned 主机内存。

        dtype 跟随模型自身，不做隐式降级 —— 否则 fp32/fp16 推理时专家池会被
        悄悄压成 bf16，产生无法解释的数值偏差。
        """
        expected = (self.num_experts, self.rank, self.hidden_dim)
        if tuple(lora_a.shape) != expected:
            raise ValueError(f"lora_A 形状应为 {expected}，实际 {tuple(lora_a.shape)}")
        if tuple(lora_b.shape) != expected:
            raise ValueError(f"lora_B 形状应为 {expected}，实际 {tuple(lora_b.shape)}")
        self.host_expert_pool = {
            "A": lora_a.detach().to("cpu", self.dtype).contiguous().pin_memory(),
            "B": lora_b.detach().to("cpu", self.dtype).contiguous().pin_memory(),
        }
        self._setup_staging()

    def _setup_staging(self) -> None:
        """常驻 staging buffer + ping-pong 事件。

        刻意不走上 caching allocator：这消除了一类真实竞态（临时张量在
        transfer_stream 上分配、在默认流上使用、随即被回收，下一轮拷贝
        可能覆写正在被 matmul 读的数据）。

        CPU 上没有 stream/event 概念，退化为单缓冲直接拷贝，
        数值结果与 GPU 路径一致，使本模块可在无显卡环境测试。
        """
        self.staging_A = [
            torch.empty(self.top_k, self.rank, self.hidden_dim,
                        device=self.device, dtype=self.dtype)
            for _ in range(self.num_staging_buffers)
        ]
        self.staging_B = [torch.empty_like(a) for a in self.staging_A]
        self._slot = 0
        self.is_cuda = torch.device(self.device).type == "cuda"
        if self.is_cuda:
            self.staging_events = [
                torch.cuda.Event() for _ in range(self.num_staging_buffers)
            ]
            cur = torch.cuda.current_stream()
            for ev in self.staging_events:      # 首拍不要阻塞
                ev.record(cur)
            self.transfer_stream = torch.cuda.Stream(device=self.device)
        else:
            self.staging_events = None
            self.transfer_stream = None

    def reset_telemetry(self) -> None:
        for t in self.telemetry.values():
            t.zero_()

    def expert_output(self, x: torch.Tensor) -> torch.Tensor:
        if not self.host_expert_pool:
            raise RuntimeError("Tier-3 专家池尚未 pin 到主机内存，请先 load_expert_pool()")
        bsz, seqlen, dim = x.shape

        last_tok = x[:, -1:, :]
        logits = self.router_little(last_tok)
        probs = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)

        self._record_telemetry(x, topv, topi)

        # 唯一一次 D2H：专家下标与权重一起取回（搬运本来就需要主机侧下标）
        info = torch.stack([topi[0, -1].to(x.dtype), topv[0, -1]]).to("cpu")
        info_list = info.tolist()
        selected = [int(v) for v in info_list[0]]
        weights = [float(v) for v in info_list[1]]

        slot = self._stage_experts(selected)
        return self._compute_experts(x, weights, slot)

    def _record_telemetry(self, x, topv, topi) -> None:
        """路由统计全部在 GPU 上累加，避免每层每 token 的 .item() 同步。"""
        t = self.telemetry
        with torch.no_grad():
            w = torch.softmax(self.router_big(x[:, -1:, :]).float(), -1)
            t["arts_weight"] += w[0, -1, 0].float()
            t["sci_weight"] += w[0, -1, 1].float()
            t["expert_calls"].index_add_(
                0, topi[0, -1], torch.ones_like(topi[0, -1])
            )

    def _stage_experts(self, selected: list[int]) -> int:
        """把 top-k 专家搬入 staging buffer。返回本拍槽位。"""
        slot = self._slot
        self._slot = (self._slot + 1) % len(self.staging_A)
        host_a = self.host_expert_pool["A"]
        host_b = self.host_expert_pool["B"]

        if not self.is_cuda:
            for j, eid in enumerate(selected):
                self.staging_A[slot][j].copy_(host_a[eid])
                self.staging_B[slot][j].copy_(host_b[eid])
            return slot

        # 等两拍前读这块 buffer 的计算结束
        self.transfer_stream.wait_event(self.staging_events[slot])
        with torch.cuda.stream(self.transfer_stream):
            for j, eid in enumerate(selected):
                self.staging_A[slot][j].copy_(host_a[eid], non_blocking=True)
                self.staging_B[slot][j].copy_(host_b[eid], non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        return slot

    def _compute_experts(
        self, x: torch.Tensor, weights: list[float], slot: int
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen
        a_buf, b_buf = self.staging_A[slot], self.staging_B[slot]

        x2d = x.reshape(n, dim)
        out = torch.zeros(n, dim, device=self.device, dtype=x.dtype)
        for j, w in enumerate(weights):
            h = torch.matmul(x2d, a_buf[j].t())
            out.addmm_(h, b_buf[j], alpha=w)

        # 标记本拍读完，供两拍后的拷贝等待
        if self.is_cuda:
            self.staging_events[slot].record(torch.cuda.current_stream())
        return out.reshape(bsz, seqlen, dim) * self.scaling


def inject_moe(model, cfg: Config, mode: str = "train", **kwargs) -> list[DualBigLittleMoE]:
    """把基座 MLP 逐层替换为双大核模块，返回注入的模块列表。"""
    cls = {"train": TrainMoE, "infer": InferMoE}[mode]
    hidden = model.config.hidden_size
    injected = []
    for layer in model.model.layers:
        layer.mlp = cls(
            layer.mlp, hidden, cfg,
            device=cfg.device, dtype=cfg.torch_dtype, **kwargs
        )
        injected.append(layer.mlp)
    return injected


__all__ = [
    "DualBigLittleMoE", "TrainMoE", "InferMoE", "RouterStats", "inject_moe",
]
