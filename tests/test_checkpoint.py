"""checkpoint 保存/加载往返与 int8 量化差分的测试（CPU 即可）。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dbl.checkpoint import (
    CHECKPOINT_KIND,
    apply_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from dbl.config import Config
from dbl.moe import inject_moe


class TinyMLP(nn.Module):
    def __init__(self, dim=16, inter=32):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class FakeLayer(nn.Module):
    """对齐真实模型的 ``layer.mlp`` 结构。"""

    def __init__(self, dim=16):
        super().__init__()
        self.mlp = TinyMLP(dim)


class FakeBase(nn.Module):
    """只提供 checkpoint 与 inject_moe 需要的接口。"""

    def __init__(self, n=3, dim=16):
        super().__init__()
        self.model = self
        self.layers = nn.ModuleList([FakeLayer(dim) for _ in range(n)])
        self.config = type("C", (), {"hidden_size": dim})()


@pytest.fixture
def cfg():
    return Config(
        model_id="dummy", num_experts=8, top_k=2, lora_rank=4, lora_alpha=8.0,
        group_sizes=(2, 2, 4), device="cpu", dtype="float32",
    )


def make_base(seed=0):
    """确定性地构造基座 —— 相同 seed 必得逐位相同的权重。

    模拟"两次从同一个 HF 仓库加载同一个基座"。
    """
    torch.manual_seed(seed)
    return FakeBase()


def build(cfg, seed=0, perturb=True, base=None):
    """构造基座 + 已注入的 MoE 模块。

    ``base`` 可传入已有基座，用于模拟"同一个基座加载两次"的真实场景
    （int8 差分要求基座逐位一致）。
    """
    if base is None:
        base = make_base(seed)
    mods = inject_moe(base, cfg, mode="train")
    if perturb:
        g = torch.Generator().manual_seed(seed)
        for m in mods:
            with torch.no_grad():
                m.lora_B.normal_(0, 0.05, generator=g)
                m.router_little.weight.normal_(0, 0.2, generator=g)
                # 模拟微调：Tier-2 相对基座产生小幅偏移
                for p in m.big_sci.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.01)
    return base, mods


class TestSaveLoadRoundtrip:
    def test_exact_roundtrip(self, cfg, tmp_path):
        base, mods = build(cfg)
        p = tmp_path / "w.pt"
        info = save_checkpoint(mods, base, cfg, p)
        assert info["num_layers"] == 3

        # 重建一份并装载，权重应逐位一致
        base2, mods2 = build(cfg, perturb=False, base=make_base())
        payload = load_checkpoint(p)
        assert payload["kind"] == CHECKPOINT_KIND
        apply_checkpoint(mods2, payload)

        for m1, m2 in zip(mods, mods2, strict=False):
            assert torch.equal(m1.big_sci.gate_proj.weight,
                               m2.big_sci.gate_proj.weight)
            assert torch.equal(m1.lora_A, m2.lora_A)
            assert torch.equal(m1.lora_B, m2.lora_B)
            assert torch.equal(m1.router_big.weight, m2.router_big.weight)
            assert torch.equal(m1.router_little.weight, m2.router_little.weight)

    def test_config_travels_with_checkpoint(self, cfg, tmp_path):
        base, mods = build(cfg)
        p = tmp_path / "w.pt"
        save_checkpoint(mods, base, cfg, p)
        payload = load_checkpoint(p)
        assert payload["cfg"].gamma == cfg.gamma
        assert payload["cfg"].num_experts == cfg.num_experts
        assert payload["cfg"].top_k == cfg.top_k
        assert payload["cfg"].group_sizes == cfg.group_sizes
        assert payload["meta"]["base_model"] == cfg.model_id

    def test_no_inference_side_constants_needed(self, cfg, tmp_path):
        """推理端应能从 checkpoint 自描述重建，无需任何手工常量。"""
        base, mods = build(cfg)
        p = tmp_path / "w.pt"
        save_checkpoint(mods, base, cfg, p)
        payload = load_checkpoint(p)
        # 结构参数全部来自 payload 自带的 config
        c = payload["cfg"]
        assert c.groups.bounds == cfg.groups.bounds
        assert c.scaling == pytest.approx(cfg.lora_alpha / cfg.lora_rank)

    def test_meta_records_steps(self, cfg, tmp_path):
        base, mods = build(cfg)
        p = tmp_path / "w.pt"
        save_checkpoint(mods, base, cfg, p, extra_meta={"steps": 500})
        payload = load_checkpoint(p)
        assert payload["meta"]["steps"] == 500


class TestInt8Delta:
    def test_smaller_than_full(self, cfg, tmp_path):
        base, mods = build(cfg)
        save_checkpoint(mods, base, cfg, tmp_path / "full.pt")
        quant = save_checkpoint(mods, base, cfg, tmp_path / "q.pt",
                                delta_dtype="int8")
        assert quant["delta_dtype"] == "int8"
        # 微型模型体积不足 0.1 MiB，按字节比较
        full_bytes = (tmp_path / "full.pt").stat().st_size
        quant_bytes = (tmp_path / "q.pt").stat().st_size
        assert quant_bytes < full_bytes, (full_bytes, quant_bytes)

    def test_reconstruction_within_tolerance(self, cfg, tmp_path):
        """同一基座下，int8 差分应高保真还原 Tier-2。"""
        base, mods = build(cfg)
        p = tmp_path / "q.pt"
        save_checkpoint(mods, base, cfg, p, delta_dtype="int8")

        # 真实场景：重新加载同一个基座
        base2, mods2 = build(cfg, perturb=False, base=make_base())
        payload = load_checkpoint(p)
        info = apply_checkpoint(mods2, payload)
        assert info["delta_dtype"] == "int8"
        # 量化误差在保存时测得并存入 meta（加载时已无 trained 权重可比）
        assert info["quant_rel_error"] is not None
        assert info["quant_rel_error"] < 0.01, info["quant_rel_error"]

        for m1, m2 in zip(mods, mods2, strict=False):
            for k, w1 in m1.big_sci.state_dict().items():
                w2 = m2.big_sci.state_dict()[k]
                err = float((w1.float() - w2.float()).abs().max())
                assert err < 0.01 * float(w1.float().abs().max()) + 1e-4

    def test_mismatched_base_is_rejected(self, cfg, tmp_path):
        """基座不一致时必须报错，不能静默还原出错误权重。"""
        base, mods = build(cfg, seed=0)
        p = tmp_path / "q.pt"
        save_checkpoint(mods, base, cfg, p, delta_dtype="int8")

        # 换一个**不同**的基座
        other, mods2 = build(cfg, seed=123, perturb=False)
        with pytest.raises(ValueError, match="基座权重与 checkpoint 不匹配"):
            apply_checkpoint(mods2, load_checkpoint(p))

    def test_experts_and_routers_unaffected_by_quantization(self, cfg, tmp_path):
        base, mods = build(cfg)
        p = tmp_path / "q.pt"
        save_checkpoint(mods, base, cfg, p, delta_dtype="int8")
        base2, mods2 = build(cfg, perturb=False, base=make_base())
        apply_checkpoint(mods2, load_checkpoint(p))
        for m1, m2 in zip(mods, mods2, strict=False):
            assert torch.equal(m1.lora_A, m2.lora_A)
            assert torch.equal(m1.lora_B, m2.lora_B)
            assert torch.equal(m1.router_little.weight, m2.router_little.weight)

    def test_missing_key_raises(self, cfg, tmp_path):
        base, mods = build(cfg)
        p = tmp_path / "q.pt"
        save_checkpoint(mods, base, cfg, p, delta_dtype="int8")
        payload = load_checkpoint(p)
        payload["state"].pop("layer_0_big_sci.gate_proj.weight")
        base2, mods2 = build(cfg, perturb=False, base=make_base())
        with pytest.raises(KeyError, match="layer_0_big_sci.gate_proj.weight"):
            apply_checkpoint(mods2, payload)


class TestLegacyCompatibility:
    def test_loads_old_flat_format(self, tmp_path):
        """旧的裸 layer_* 字典应仍可识别（但标记为 legacy）。"""
        legacy = {
            "layer_0_big_sci": {"gate_proj.weight": torch.zeros(4, 4)},
            "layer_0_lora_A": torch.zeros(8, 4, 16),
        }
        p = tmp_path / "old.pt"
        torch.save(legacy, p)
        payload = load_checkpoint(p)
        assert payload["kind"] == "legacy"
        assert payload["cfg"] is None
        assert "layer_0_lora_A" in payload["state"]
