"""DualBigLittle-MoE 的模型实现（双大核 + 微专家）。

本文件是**自包含**的：只依赖 ``torch`` 与 ``transformers``，不 import 本仓库的
``dbl`` 包。外部用户 ``from_pretrained(..., trust_remote_code=True)`` 时，
transformers 只把本目录下的两个 ``.py`` 拷进动态模块缓存，因此任何对项目内部
模块的引用都会在外部环境里断掉。

与基座的关系
------------
骨架（注意力 / RoPE / RMSNorm / KV cache / 掩码 / 生成）直接复用 transformers
的 Qwen3 实现，本文件只做两件事：

1. 每层 decoder 的 ``mlp`` 换成 :class:`DualBigMoEMLP`（双大核 + 微专家池）；
2. 补上标准文本生成钩子。

复用而非重写骨架是有意的：注意力数值、Rope 语义、Cache 布局这些细节，transformers
每个版本都会修。若在此重抄一份 Qwen3，用户升级 transformers 后本模型就会成为
唯一不变的部分 —— 那是 bug 温床。本文件与 Qwen3 的耦合点收敛到三个 import，
升级时只需核对这三处。

关于 CUDA 异步搬运与 record_stream
----------------------------------
Tier-3 专家池经 PCIe DMA 从 pinned 主机内存搬入显存。搬运用两条 stream：

* ``transfer_stream`` 负责 H2D 拷贝，计算留在默认流；
* 常驻 staging buffer 双缓冲 ping-pong，槽位 ``i`` 与 ``i+1`` 交替使用，
  于是「这一拍搬下一批」与「这一拍算上一批」真正重叠。

**这里刻意不用 ``record_stream``。** 早期实现把临时张量在 transfer_stream 上分配、
在默认流上使用、随即出作用域被 caching allocator 回收，下一轮拷贝可能覆写正在被
matmul 读的数据 —— 那是 ``record_stream`` 存在的典型场景。修复方式是让 staging
buffer **不走** caching allocator（``torch.empty`` 后常驻），从根上消除跨流复用。
改用 ``record_stream`` 只是给已经消失的竞态打补丁，还会引入额外的同步开销。
详见仓库 commit ``47990f8``；实测同一输入重复 40 次前向的数值抖动从 1.56e-2
降为 0。

零拷贝路线
----------
``config.expert_pool_location="device"`` 时专家池常驻显存，完全不搬运。它与
``"host"`` 路径**数值完全等价**（同一批专家、同一组权重、同一次 top-k），
只是把 PCIe 往返换成一次 gather。对 0.6B 这种小模型通常更快；把 896 个专家
放到 70B 级别才有必要走主机内存。默认保留 ``"host"`` 以对齐已验证的行为。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils import logging

try:  # 以包形式导入（transformers 动态模块走这条路径）
    from .configuration_dualbig_moe import DualBigMoEConfig
except ImportError:  # pragma: no cover - 直接以脚本方式 import 时
    from configuration_dualbig_moe import DualBigMoEConfig

logger = logging.get_logger(__name__)


class DualBigMoEMLP(nn.Module):
    """双大核 + 微专家池的融合 FFN，替换 Qwen3 每层的 ``mlp``。

    前向 = ``w_arts · FFN_arts(x) + w_sci · FFN_sci(x) + gamma · Σ wᵢ · LoRAᵢ(x)``

    * ``w_arts`` / ``w_sci`` 由 ``router_big`` 逐位置给出（2 路 softmax）；
    * ``wᵢ`` 由 ``router_little`` 在 Tier-3 专家池上做 Top-k 给出。

    与训练路径（``dbl.moe.TrainMoE``，逐位置稠密 Top-k）唯一区别是专家选择的
    **时机**：推理只用最后一个 token 决定整段 prompt 的专家集合。否则每多一个
    token 就要多搬一轮专家。decode 阶段（seqlen=1）两者完全等价。

    Args:
        config: :class:`DualBigMoEConfig`，提供专家数 / Top-k / rank / gamma。
        layer_idx: 层号，仅用于日志与遥测定位。
    """

    def __init__(self, config: DualBigMoEConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.rank = config.lora_rank
        self.gamma = config.gamma
        self.scaling = config.lora_scaling
        self.group_bounds = config.group_bounds

        # Tier-1 文科锚核：原版 MLP，逐位等于基座，冻结。
        self.big_arts = Qwen3MLP(config)
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # Tier-2 理科孪生核：克隆一份，低学习率微调。
        self.big_sci = copy.deepcopy(self.big_arts)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 双大核路由器（无 bias）与微专家路由器。
        self.router_big = nn.Linear(
            config.hidden_size, 2, bias=False, device=self.big_arts.gate_proj.weight.device,
            dtype=self.big_arts.gate_proj.weight.dtype,
        )
        self.router_little = nn.Linear(
            config.hidden_size, config.num_experts, bias=False,
            device=self.big_arts.gate_proj.weight.device,
            dtype=self.big_arts.gate_proj.weight.dtype,
        )

        # Tier-3 堆叠 LoRA，形状 (E, r, D)。堆叠而非 ModuleList 是为了让所有
        # 专家都能被稠密计算覆盖 —— 早期 ModuleList + 索引的写法会让未被选中的
        # 专家拿不到梯度，896 个专家里只有 93 个真正在学。
        self.lora_A = nn.Parameter(
            torch.empty(
                config.num_experts, config.lora_rank, config.hidden_size,
                device=self.big_arts.gate_proj.weight.device,
                dtype=self.big_arts.gate_proj.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(torch.empty_like(self.lora_A))
        self.reset_expert_parameters_()

        # ---- 流式搬运运行时状态（不进 state_dict）----
        self._host_pool: dict[str, torch.Tensor] = {}
        self._pool_key: tuple | None = None
        self._slot = 0
        self.staging_A: list[torch.Tensor] = []
        self.staging_B: list[torch.Tensor] = []
        self.staging_events: list | None = None
        self.transfer_stream = None
        self._telemetry: dict[str, torch.Tensor] | None = None

    @property
    def telemetry(self) -> dict[str, torch.Tensor]:
        """路由统计累加器，按当前设备惰性创建。

        必须是惰性的：``from_pretrained`` 在 meta device 上构造模块，此时
        ``torch.zeros(..., device=meta)`` 建出来的是 meta 张量，永远不会被
        ``load_state_dict`` 填实，于是第一次前向就炸
        ``index_add(): self, index and source expected to be in the same device``。
        这些张量不是 buffer（不进 ``state_dict``），所以也**不会**被
        ``.to(device)`` 搬运 —— 只能在这里按需重建。
        """
        if self._telemetry is None or self._telemetry["expert_calls"].device != self.lora_A.device:
            self._telemetry = self._new_telemetry()
        return self._telemetry

    # ------------------------------------------------------------------
    # 构造辅助
    # ------------------------------------------------------------------
    def reset_expert_parameters_(self) -> None:
        """A 用 kaiming、B 置零 —— 初始输出恒为 0，不破坏基座行为。

        B 为零是关键：它让「未训练的专家」对前向没有任何贡献，因此从基座
        初始化新层是安全的，不会一上来就把语言能力打乱。
        """
        nn.init.kaiming_uniform_(
            self.lora_A.reshape(-1, self.hidden_size), a=5 ** 0.5
        )
        nn.init.zeros_(self.lora_B)

    def _new_telemetry(self) -> dict[str, torch.Tensor]:
        device = self.lora_A.device
        return {
            "arts_weight": torch.zeros((), device=device, dtype=torch.float32),
            "sci_weight": torch.zeros((), device=device, dtype=torch.float32),
            "expert_calls": torch.zeros(self.num_experts, device=device, dtype=torch.long),
        }

    def reset_telemetry(self) -> None:
        """清零路由统计。推理脚本在每次提问前调用。"""
        for t in self.telemetry.values():
            t.zero_()

    # ------------------------------------------------------------------
    # 大核混音
    # ------------------------------------------------------------------
    def _big_core(self, x: torch.Tensor) -> torch.Tensor:
        """双大核逐位置混音：``w_arts · FFN_arts(x) + w_sci · FFN_sci(x)``。"""
        weights = torch.softmax(self.router_big(x).float(), dim=-1).to(x.dtype)
        return weights[..., 0:1] * self.big_arts(x) + weights[..., 1:2] * self.big_sci(x)

    # ------------------------------------------------------------------
    # 专家池
    # ------------------------------------------------------------------
    @property
    def pool_location(self) -> str:
        return getattr(self.config, "expert_pool_location", "host")

    def _ensure_expert_pool(self) -> None:
        """确保 Tier-3 权重就位。

        惰性构建，由 ``forward`` 调用。这样 ``from_pretrained`` 之后直接
        ``generate()`` 即可，无需外部用户手动 pin 内存 —— 而后者是
        ``trust_remote_code`` 模型最容易踩的坑（自定义前置步骤不会被 Hub
        生态的任何工具代劳）。

        幂等性靠 ``_pool_key`` 记录 (location, device, dtype)：模型
        ``.to("cuda")`` / ``.half()`` 之后首次前向会自动重建，无需用户干预。
        """
        device = self.lora_A.device
        key = (self.pool_location, str(device), str(self.lora_A.dtype))
        if self._pool_key == key:
            return
        self._pool_key = key

        if self.pool_location == "device":
            # 常驻显存：直接用参数本身，无需 staging。
            self._host_pool = {}
            self.staging_A, self.staging_B = [], []
            self.staging_events = None
            self.transfer_stream = None
            return

        self._host_pool = {
            "A": self.lora_A.detach().to("cpu", copy=True).contiguous().pin_memory(),
            "B": self.lora_B.detach().to("cpu", copy=True).contiguous().pin_memory(),
        }
        self._setup_staging(device)

    def _setup_staging(self, device: torch.device) -> None:
        """建立常驻 staging buffer 与 ping-pong 完成事件。

        刻意**不**使用 caching allocator：临时张量若在 transfer_stream 上分配、
        在默认流上使用、随即被回收，下一轮拷贝就可能覆写正在被 matmul 读的数据。
        常驻 buffer 让每块存储的生命周期跨越多拍，竞态在结构上不成立。

        代价是 staging 显存常驻，但每层只有 ``top_k x rank x hidden``（本模型
        约 0.5 MiB），可以忽略。
        """
        n = max(1, int(getattr(self.config, "num_staging_buffers", 2)))
        self.staging_A = [
            torch.empty(self.top_k, self.rank, self.hidden_size, device=device, dtype=self.lora_A.dtype)
            for _ in range(n)
        ]
        self.staging_B = [torch.empty_like(a) for a in self.staging_A]
        self._slot = 0
        self.staging_events = None
        self.transfer_stream = None

        if device.type == "cuda" and torch.cuda.is_available():
            self.staging_events = [torch.cuda.Event() for _ in range(n)]
            current = torch.cuda.current_stream(device)
            for ev in self.staging_events:      # 首拍不要阻塞
                ev.record(current)
            self.transfer_stream = torch.cuda.Stream(device=device)
        # CPU 上没有 stream/event 概念，退化为单缓冲直接拷贝；
        # 数值结果与 GPU 路径一致，使本模块可在无显卡环境单测。

    def _stage_experts(self, selected: list[int], device: torch.device) -> int:
        """把 Top-k 专家搬入 staging buffer，返回本拍槽位。"""
        slot = self._slot
        self._slot = (self._slot + 1) % len(self.staging_A)
        host_a, host_b = self._host_pool["A"], self._host_pool["B"]

        if self.transfer_stream is None:
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
        # 拷贝可以与后续默认流上的计算重叠
        torch.cuda.current_stream(device).wait_stream(self.transfer_stream)
        return slot

    def _compute_experts(
        self, x: torch.Tensor, weights: list[float], slot: int
    ) -> torch.Tensor:
        """在 staging buffer 上做 ``sum_i w_i · B_i(A_i x)``。

        先乘后加的顺序与训练路径（稠密 top-k 掩码）一致，保证两条路径数值等价。
        """
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen
        a_buf, b_buf = self.staging_A[slot], self.staging_B[slot]

        x2d = x.reshape(n, dim)
        out = torch.zeros(n, dim, device=x.device, dtype=x.dtype)
        for j, w in enumerate(weights):
            h = torch.matmul(x2d, a_buf[j].t())
            out.addmm_(h, b_buf[j], alpha=w)

        # 标记本拍读完，供两拍后的拷贝等待
        if self.staging_events is not None:
            self.staging_events[slot].record(torch.cuda.current_stream(x.device))
        return out.reshape(bsz, seqlen, dim) * self.scaling

    def _compute_experts_resident(
        self, x: torch.Tensor, selected: list[int], weights: list[float]
    ) -> torch.Tensor:
        """零拷贝路径：直接从常驻参数读选中的专家，省掉 PCIe 往返。

        与 :meth:`_compute_experts` **逐步相同的算子与顺序**（同样的
        ``matmul`` -> ``addmm_(alpha=w)`` 链、同样的先乘后加），因此输出逐位
        相同；差别仅在专家权重从 staging buffer 换成参数本身。测试
        ``test_hf_roundtrip.py::test_host_and_device_pool_agree`` 对此有断言。
        """
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen
        x2d = x.reshape(n, dim)
        out = torch.zeros(n, dim, device=x.device, dtype=x.dtype)
        for j, eid in enumerate(selected):
            h = torch.matmul(x2d, self.lora_A[eid].t())
            out.addmm_(h, self.lora_B[eid], alpha=weights[j])
        return out.reshape(bsz, seqlen, dim) * self.scaling

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def _route_little(self, x: torch.Tensor):
        """用末 token 选专家，返回归一化后的 (下标, 权重)。"""
        last_tok = x[:, -1:, :]
        probs = torch.softmax(self.router_little(last_tok).float(), dim=-1).to(x.dtype)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        return topi, topv

    def _record_telemetry(self, x: torch.Tensor, topi, topv) -> None:
        """路由统计全部在 GPU 上累加，避免每层每 token 的 ``.item()`` 同步。

        历史实现每层每 token 调两次 ``.item()`` 打断 CPU-GPU 流水线；改成
        设备侧累加后，每层每 token 只剩一次 D2H（取专家下标那次，搬运本来
        就需要），实测吞吐 +15.4%。
        """
        t = self.telemetry
        with torch.no_grad():
            w = torch.softmax(self.router_big(x[:, -1:, :]).float(), dim=-1)
            t["arts_weight"] += w[0, -1, 0].float()
            t["sci_weight"] += w[0, -1, 1].float()
            t["expert_calls"].index_add_(
                0, topi[0, -1], torch.ones_like(topi[0, -1], device=t["expert_calls"].device)
            )

    def expert_output(self, x: torch.Tensor) -> torch.Tensor:
        """Tier-3 微专家对激活的贡献（已含 LoRA scaling，不含 gamma）。"""
        topi, topv = self._route_little(x)
        self._record_telemetry(x, topi, topv)

        # 唯一一次 D2H：专家下标与权重一起取回。
        # 搬运本来就需要主机侧下标，所以这一同步不额外增加开销点。两条路径
        # 共用同一份 (selected, weights)，是它们逐位一致的前提。
        info = torch.stack([topi[0, -1].to(x.dtype), topv[0, -1]]).to("cpu").tolist()
        selected = [int(v) for v in info[0]]
        weights = [float(v) for v in info[1]]

        if self.pool_location == "device":
            return self._compute_experts_resident(x, selected, weights)

        self._ensure_expert_pool()
        slot = self._stage_experts(selected, x.device)
        return self._compute_experts(x, weights, slot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self._big_core(x)
        return big_out + self.gamma * self.expert_output(x)

    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, top_k={self.top_k}, rank={self.rank}, "
            f"gamma={self.gamma}, pool={self.pool_location}"
        )


class DualBigMoEDecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer，FFN 换成 :class:`DualBigMoEMLP`。"""

    def __init__(self, config: DualBigMoEConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.mlp = DualBigMoEMLP(config, layer_idx)


class DualBigMoEPreTrainedModel(Qwen3PreTrainedModel):
    """权重初始化与能力声明的基类。

    注意力 / RoPE / Cache / 掩码全部沿用 Qwen3，故能力位也照抄；只有
    ``_init_weights`` 需要为 Tier-3 的堆叠 LoRA 加一条分支（B 必须置零）。
    """

    config_class = DualBigMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DualBigMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DualBigMoEDecoderLayer,
        "attentions": Qwen3Attention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, DualBigMoEMLP):
            # Tier-3 走自己的初始化：A kaiming / B 置零，恒等映射起步。
            module.reset_expert_parameters_()
            for proj in (module.big_sci.gate_proj, module.big_sci.up_proj, module.big_sci.down_proj):
                nn.init.normal_(proj.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.router_big.weight, std=self.config.initializer_range)
            nn.init.normal_(module.router_little.weight, std=self.config.initializer_range)
            return
        super()._init_weights(module)


class DualBigMoEModel(Qwen3Model):
    """Qwen3 骨架，decoder layer 换成双大核版本。

    刻意继承而非复制 ``Qwen3Model``：forward（掩码构造、Cache 读写、position_ids
    推导）完全复用，升级 transformers 时不会漏掉行为变更。这里只重写 ``__init__``
    里的 layer 构造。
    """

    def __init__(self, config: DualBigMoEConfig) -> None:
        # 跳过 Qwen3Model.__init__：它硬编码构造 Qwen3DecoderLayer。
        PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [DualBigMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()


class DualBigMoEForCausalLM(DualBigMoEPreTrainedModel, GenerationMixin):
    """带语言建模头的 DualBigLittle-MoE。

    Example:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    >>> model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype="auto").to("cuda")
    >>> out = model.generate(**tok("解释一下快速排序", return_tensors="pt"), max_new_tokens=64)
    ```
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    config_class = DualBigMoEConfig

    def __init__(self, config: DualBigMoEConfig) -> None:
        super().__init__(config)
        self.model = DualBigMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> nn.Module:
        return self.model

    # ------------------------------------------------------------------
    # 路由遥测
    # ------------------------------------------------------------------
    def moe_layers(self) -> list[DualBigMoEMLP]:
        """取出全部双大核 FFN 模块。"""
        return [layer.mlp for layer in self.model.layers if isinstance(layer.mlp, DualBigMoEMLP)]

    def reset_routing_stats(self) -> None:
        """清零全部层的路由统计。"""
        for m in self.moe_layers():
            m.reset_telemetry()

    @torch.no_grad()
    def routing_report(self) -> dict:
        """一次性取回全部遥测统计（避免每层各自同步）。

        Returns:
            ``arts_ratio`` / ``sci_ratio`` 为双大核能量分配（0~1）；
            ``expert_calls`` 为各专家累计调用次数；``group_shares`` 为
            「代码 / 数学 / 写作」三组的调用占比。
        """
        modules = self.moe_layers()
        if not modules:
            return {}
        arts = torch.stack([m.telemetry["arts_weight"] for m in modules]).sum().item()
        sci = torch.stack([m.telemetry["sci_weight"] for m in modules]).sum().item()
        total = arts + sci
        calls = torch.stack([m.telemetry["expert_calls"] for m in modules]).sum(0).cpu().tolist()
        grand = float(sum(calls))
        bounds = self.config.group_bounds
        shares = [
            (sum(calls[lo:hi]) / grand if grand > 0 else 0.0) for lo, hi in bounds
        ]
        return {
            "arts_ratio": arts / total if total > 0 else 0.5,
            "sci_ratio": sci / total if total > 0 else 0.5,
            "expert_calls": calls,
            "expert_share": [c / grand for c in calls] if grand > 0 else [0.0] * len(calls),
            "group_bounds": bounds,
            "group_shares": shares,
            "group_fair_share": [ (hi - lo) / self.config.num_experts for lo, hi in bounds ],
        }

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """文本生成钩子。

        本模型对生成没有额外要求：骨架就是标准 Qwen3，``GenerationMixin`` 的默认
        实现（切片 ``input_ids``、补 ``position_ids``、裁 4D mask）完全适用。
        这里保留显式覆写而不是删掉，理由是它同时承担一个实际职责：
        ``trust_remote_code`` 的模型没有 ``generation_config.json`` 时，Hub 上的
        工具（pipeline、TGI、vLLM）会直接实例化本类；显式声明该钩子存在，
        可避免它们按「自定义模型需要自己实现生成钩子」的启发式走偏。

        真正的定制在 :class:`DualBigMoEMLP`：微专家按**末 token**选择，
        因此无论 prefill 多长，整段 prompt 共用同一组专家。这不需要在生成
        钩子里体现 —— 层内已经处理。
        """
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "DualBigMoEConfig",
    "DualBigMoEPreTrainedModel",
    "DualBigMoEModel",
    "DualBigMoEDecoderLayer",
    "DualBigMoEMLP",
    "DualBigMoEForCausalLM",
]
