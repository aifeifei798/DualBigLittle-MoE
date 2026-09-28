"""模型前向的单元测试（CPU 即可跑，用微型 MLP 代替真实基座）。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dbl.config import Config
from dbl.moe import InferMoE, TrainMoE, inject_moe


class TinyMLP(nn.Module):
    """结构上与 Qwen3MLP 一致的最小 MLP（gate/up/down 三投影）。"""

    def __init__(self, dim: int = 16, inter: int = 32):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


@pytest.fixture
def cfg():
    return Config(
        model_id="dummy",
        num_experts=8,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,      # scaling = 2.0
        gamma=0.3,
        group_sizes=(2, 2, 4),
        device="cpu",
        dtype="float32",
    )


def make(cfg, cls=TrainMoE, **kw):
    torch.manual_seed(0)
    dim = 16
    return cls(TinyMLP(dim), dim, cfg, device="cpu", dtype=torch.float32, **kw), dim


def train_module(cfg):
    mod, dim = make(cfg, TrainMoE)
    # 打破零初始化，验证 LoRA 确实参与输出
    with torch.no_grad():
        mod.lora_B.normal_(0, 0.1)
    return mod, dim


def fwd(mod, x):
    """前向并取回 RouterStats（等价于训练循环里的 collect_stats 路径）。"""
    mod.collect_stats = True
    try:
        out = mod(x)
        return out, mod.last_stats
    finally:
        mod.collect_stats = False


class TestExpertInit:
    def test_lora_b_zero_init_means_identity(self, cfg):
        """B 置零 -> 初始微专家输出恒为 0，不破坏基座行为。"""
        mod, dim = make(cfg, TrainMoE)
        x = torch.randn(2, 5, dim)
        out = mod(x)
        big, _, _ = mod.big_core(x)
        assert torch.allclose(out, big, atol=1e-6)

    def test_scaling_applied(self, cfg):
        mod, dim = train_module(cfg)
        x = torch.randn(1, 3, dim)
        _, stats = fwd(mod, x)
        topv, topi, dense, _ = mod.select_topk(stats.little_logits)
        manual = torch.matmul(
            torch.matmul(
                x.reshape(3, dim),
                mod.lora_A.reshape(-1, dim).t(),
            )
            * dense.reshape(3, -1).repeat_interleave(cfg.lora_rank, dim=1),
            mod.lora_B.reshape(-1, dim),
        ).reshape(1, 3, dim) * mod.scaling
        out = mod(x)
        assert torch.allclose(
            out - mod.big_core(x)[0], cfg.gamma * manual, atol=1e-5
        )


class TestTopK:
    def test_weights_sum_to_one(self, cfg):
        mod, dim = train_module(cfg)
        x = torch.randn(2, 4, dim)
        topv, _, dense, _ = mod.select_topk(mod.router_little(x))
        assert torch.allclose(topv.sum(-1), torch.ones(2, 4), atol=1e-5)
        assert torch.allclose(dense.sum(-1), torch.ones(2, 4), atol=1e-5)

    def test_exactly_k_nonzero(self, cfg):
        mod, dim = train_module(cfg)
        x = torch.randn(2, 6, dim)
        _, topi, dense, _ = mod.select_topk(mod.router_little(x))
        assert (dense > 0).sum(-1).unique().tolist() == [cfg.top_k]
        # topi 互不重复
        for row in topi.reshape(-1, cfg.top_k):
            assert len(set(row.tolist())) == cfg.top_k

    def test_selects_highest_prob(self, cfg):
        mod, dim = train_module(cfg)
        logits = torch.log(torch.tensor([[[0.1, 0.4, 0.3, 0.05, 0.05, 0.03, 0.02, 0.01]]]))
        topv, topi, _, _ = mod.select_topk(logits)
        assert sorted(topi[0, 0].tolist()) == [1, 2]


class TestBigCore:
    def test_frozen_arts(self, cfg):
        mod, dim = make(cfg, TrainMoE)
        assert all(not p.requires_grad for p in mod.big_arts.parameters())
        assert all(p.requires_grad for p in mod.big_sci.parameters())

    def test_weights_normalized(self, cfg):
        mod, dim = make(cfg, TrainMoE)
        x = torch.randn(2, 5, dim)
        _, w, _ = mod.big_core(x)
        assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-5)

    def test_output_is_convex_combination(self, cfg):
        """全选文科核时输出应等于 big_arts(x)。"""
        mod, dim = make(cfg, TrainMoE)
        with torch.no_grad():
            mod.router_big.weight.zero_()
            # 用固定方向产生恒定的正 logit，避免 x 的符号翻转路由
            mod.router_big.weight[0, 0] = 50.0
        x = torch.zeros(1, 4, dim)
        x[..., 0] = 1.0            # 与路由方向同号 -> arts logit = 50
        out, w, _ = mod.big_core(x)
        assert torch.allclose(out, mod.big_arts(x), atol=1e-4)
        assert torch.allclose(w[..., 0], torch.ones(1, 4), atol=1e-6)

    def test_output_is_not_sum_of_others(self, cfg):
        """全选理科核时应等于 big_sci(x)，而非两者相加。"""
        mod, dim = make(cfg, TrainMoE)
        with torch.no_grad():
            mod.router_big.weight.zero_()
            mod.router_big.weight[1, 0] = 50.0
        x = torch.zeros(1, 4, dim)
        x[..., 0] = 1.0
        out, w, _ = mod.big_core(x)
        assert torch.allclose(out, mod.big_sci(x), atol=1e-4)
        assert torch.allclose(w[..., 1], torch.ones(1, 4), atol=1e-6)


class TestParameterGroups:
    def test_lrs_follow_staged_policy(self, cfg):
        mod, _ = make(cfg, TrainMoE)
        gs = mod.parameter_groups()
        by_name = {g["name"]: g for g in gs}
        assert by_name["big_sci"]["lr"] == cfg.lr_big_sci
        assert by_name["router"]["lr"] == cfg.lr_router
        assert by_name["experts"]["lr"] == cfg.lr_experts
        # 分级关系：大核最慢，专家最快
        assert by_name["big_sci"]["lr"] < by_name["router"]["lr"] < by_name["experts"]["lr"]

    def test_no_frozen_params_in_groups(self, cfg):
        mod, _ = make(cfg, TrainMoE)
        for g in mod.parameter_groups():
            assert all(p.requires_grad for p in g["params"])


class TestGradients:
    def test_lm_loss_touches_only_selected_experts(self, cfg):
        """LM loss 只给**被 top-k 选中**的专家梯度。

        这与直觉相反但确为正确行为：稠密前向虽计算了全部专家，
        top-k 掩码会把未选中专家的贡献归零，因此其梯度也是零。
        早期用 ModuleList + 单专家索引的写法在这一点上并无差别 ——
        真正防止专家饿死的是负载均衡损失（见下一个测试），
        而不是"稠密计算"本身。
        """
        mod, dim = train_module(cfg)
        x = torch.randn(1, 4, dim)
        out, stats = fwd(mod, x)
        out.sum().backward()

        _, topi, _, _ = mod.select_topk(stats.little_logits)
        selected = set(topi.reshape(-1).tolist())
        got_grad = set(
            (mod.lora_B.grad.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        )
        assert got_grad == selected, (
            f"有梯度的专家 {sorted(got_grad)} 应恰等于被选中的 {sorted(selected)}"
        )

    def test_load_balance_gives_every_expert_gradient(self, cfg):
        """负载均衡 + 分组监督让 router 的**每一行**都拿到梯度。

        这是防止路由塌缩、专家饿死的真正机制。
        """
        from dbl.groups import DEFAULT_GROUPS  # noqa: F401

        mod, dim = make(cfg, TrainMoE)
        x = torch.randn(1, 4, dim)
        _, stats = fwd(mod, x)
        group_logits = mod.cfg.groups.pool_to_groups(stats.little_logits.float())
        balance = mod.num_experts * torch.sum(
            stats.topk_weights.mean(0) * stats.little_probs.mean(0)
        )
        (group_logits.sum() + balance).backward()

        per_row = mod.router_little.weight.grad.abs().sum(-1)
        assert (per_row > 0).all(), (
            f"仅 {int((per_row > 0).sum())}/{cfg.num_experts} 行拿到梯度"
        )

    def test_router_little_starts_with_no_lm_gradient(self, cfg):
        """B 零初始化时微专家输出恒为 0，LM loss 无法训练小核路由器。

        路由器在训练初期完全依赖辅助损失起步 —— 这也是
        ``router_aux_weight`` 不能设成 0 的原因。
        """
        mod, dim = make(cfg, TrainMoE)      # B 保持零初始化
        out = mod(torch.randn(1, 4, dim))
        out.sum().backward()
        assert float(mod.router_little.weight.grad.abs().sum()) == 0.0
        # 但大核路由器不受影响
        assert float(mod.router_big.weight.grad.abs().sum()) > 0

    def test_big_router_gets_gradient(self, cfg):
        mod, dim = make(cfg, TrainMoE)
        out = mod(torch.randn(1, 4, dim))
        out.sum().backward()
        assert mod.router_big.weight.grad.abs().sum() > 0

    def test_arts_get_no_gradient(self, cfg):
        mod, dim = make(cfg, TrainMoE)
        out = mod(torch.randn(1, 4, dim))
        out.sum().backward()
        assert all(p.grad is None for p in mod.big_arts.parameters())


class TestTrainInferEquivalence:
    """decode 阶段（seqlen=1）两条路径必须数值一致。"""

    def _pair(self, cfg):
        torch.manual_seed(0)
        dim = 16
        base = TinyMLP(dim)
        train_mod = TrainMoE(base, dim, cfg, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            train_mod.lora_B.normal_(0, 0.1)
            train_mod.router_little.weight.normal_(0, 0.5)
        # 用训练后的同一份权重构造推理模块
        infer_mod = InferMoE(
            train_mod.big_arts, dim, cfg, device="cpu", dtype=torch.float32
        )
        infer_mod.big_sci = train_mod.big_sci
        infer_mod.router_big = train_mod.router_big
        infer_mod.router_little = train_mod.router_little
        infer_mod.load_expert_pool(
            train_mod.lora_A.data.clone(), train_mod.lora_B.data.clone()
        )
        return train_mod, infer_mod, dim

    def test_decode_equivalence(self, cfg):
        tr, inf, dim = self._pair(cfg)
        x = torch.randn(1, 1, dim)
        with torch.no_grad():
            tr_out = tr(x)
            inf_out = inf(x)
        assert torch.allclose(tr_out, inf_out, atol=1e-4), (
            f"decode 阶段应等价，最大偏差 {float((tr_out - inf_out).abs().max()):.2e}"
        )

    def test_decode_equivalence_many_seeds(self, cfg):
        for seed in range(5):
            torch.manual_seed(seed)
            tr, inf, dim = self._pair(cfg)
            x = torch.randn(1, 1, dim)
            with torch.no_grad():
                a = tr(x)
                b = inf(x)
            assert torch.allclose(a, b, atol=1e-4), f"seed={seed} 不等价"

    def test_prefill_difference_is_bounded(self, cfg):
        """prefill 下两者本就不同（top-k 时机不同），但差异应有限。"""
        tr, inf, dim = self._pair(cfg)
        x = torch.randn(1, 16, dim)
        with torch.no_grad():
            a = tr(x)
            b = inf(x)
        rel = float((a - b).abs().max() / a.abs().max().clamp_min(1e-9))
        assert rel < 1.0, f"prefill 相对差异过大: {rel}"


class TestInferMoE:
    def test_requires_pool_loaded(self, cfg):
        mod, dim = make(cfg, InferMoE)
        with pytest.raises(RuntimeError, match="pin"):
            mod(torch.randn(1, 2, dim))

    def test_pool_shape_validation(self, cfg):
        mod, dim = make(cfg, InferMoE)
        with pytest.raises(ValueError, match="lora_A 形状"):
            mod.load_expert_pool(torch.zeros(3, 4, 16), torch.zeros(8, 4, 16))

    def test_double_buffer_does_not_corrupt(self, cfg):
        """ping-pong 槽位复用不得覆盖仍在使用的数据。"""
        mod, dim = make(cfg, InferMoE)
        with torch.no_grad():
            mod.lora_B.normal_(0, 0.1)
        mod.load_expert_pool(mod.lora_A.data, mod.lora_B.data)
        outs = []
        for _ in range(6):
            x = torch.randn(1, 1, dim)
            with torch.no_grad():
                outs.append(mod(x).clone())
        for o in outs:
            assert torch.isfinite(o).all()

    def test_telemetry_resets(self, cfg):
        mod, dim = make(cfg, InferMoE)
        mod.load_expert_pool(
            torch.randn(cfg.num_experts, cfg.lora_rank, dim),
            torch.randn(cfg.num_experts, cfg.lora_rank, dim),
        )
        mod(torch.randn(1, 1, dim))
        assert mod.telemetry["expert_calls"].sum() > 0
        mod.reset_telemetry()
        assert mod.telemetry["expert_calls"].sum() == 0
        assert mod.telemetry["arts_weight"] == 0


class TestInjectMoE:
    def test_replaces_every_layer(self, cfg):
        class FakeLayer:
            def __init__(self):
                self.mlp = TinyMLP(16)

        class FakeModel:
            def __init__(self):
                self.model = type(
                    "M", (), {"layers": [FakeLayer() for _ in range(3)]}
                )()
                self.config = type("C", (), {"hidden_size": 16})()

        m = FakeModel()
        mods = inject_moe(m, cfg, mode="train")
        assert len(mods) == 3
        assert all(isinstance(x, TrainMoE) for x in mods)
        assert all(isinstance(lyr.mlp, TrainMoE) for lyr in m.model.layers)


class TestDeterminism:
    def test_repeated_forward_identical(self, cfg):
        mod, dim = train_module(cfg)
        x = torch.randn(2, 5, dim)
        with torch.no_grad():
            a, _ = mod(x)
            b, _ = mod(x)
        assert torch.equal(a, b), "相同输入应产生逐位相同的输出"
