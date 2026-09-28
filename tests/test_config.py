"""分组拓扑与配置的单元测试（无需 GPU / 模型下载）。"""

from __future__ import annotations

import math

import pytest
import torch

from dbl.config import CONFIG_VERSION, Config
from dbl.groups import DEFAULT_GROUPS, ExpertGroups


class TestExpertGroups:
    def test_default_topology(self):
        g = DEFAULT_GROUPS
        assert g.num_experts == 32
        assert g.sizes == (8, 8, 16)
        assert g.num_groups == 3
        # 边界必须连续覆盖 0..32
        assert g.bounds[0][0] == 0 and g.bounds[-1][1] == 32

    def test_bounds_must_be_contiguous(self):
        with pytest.raises(ValueError, match="不连续"):
            ExpertGroups(((0, 8), (9, 16), (16, 32)), 32)

    def test_must_cover_all_experts(self):
        with pytest.raises(ValueError, match="覆盖"):
            ExpertGroups(((0, 8), (8, 16)), 32)

    def test_must_start_at_zero(self):
        with pytest.raises(ValueError, match="从 0 开始"):
            ExpertGroups(((1, 8), (8, 16), (16, 32)), 32)

    def test_group_of_expert(self):
        g = DEFAULT_GROUPS
        assert g.group_of_expert(0) == 0
        assert g.group_of_expert(7) == 0
        assert g.group_of_expert(8) == 1
        assert g.group_of_expert(15) == 1
        assert g.group_of_expert(16) == 2
        assert g.group_of_expert(31) == 2

    def test_group_of_expert_out_of_range(self):
        with pytest.raises(ValueError, match="越界"):
            DEFAULT_GROUPS.group_of_expert(32)

    def test_domain_mapping(self):
        g = DEFAULT_GROUPS
        assert g.group_of_domain("Code") == 0
        assert g.group_of_domain("Math") == 1
        assert g.group_of_domain("Arts") == 2
        with pytest.raises(ValueError, match="未知领域"):
            g.group_of_domain("Biology")

    def test_from_sizes_matches_bounds(self):
        g = ExpertGroups.from_sizes((8, 8, 16))
        assert g.bounds == ((0, 8), (8, 16), (16, 32))

    def test_log_size_correction(self):
        g = DEFAULT_GROUPS
        assert g.log_size_correction() == (
            math.log(8), math.log(8), math.log(16)
        )


class TestGroupPooling:
    """分组级路由池化：``logsumexp - log(size)`` 的规模归一。"""

    def test_pooling_shape(self):
        logits = torch.randn(4, 10, 32)
        pooled = DEFAULT_GROUPS.pool_to_groups(logits)
        assert pooled.shape == (4, 10, 3)

    def test_equal_probability_within_group_is_group_mean(self):
        """组内 logits 全相等时，池化值应等于该组的 log(size)。"""
        logits = torch.zeros(1, 1, 32)
        pooled = DEFAULT_GROUPS.pool_to_groups(logits)
        # logsumexp(zeros over n) = log(n); 减去 log(n) 后为 0
        assert torch.allclose(pooled, torch.zeros(1, 1, 3), atol=1e-6)

    def test_correction_removes_large_group_bias(self):
        """关键性质：不做 ``-log(size)`` 修正时大组会被系统性高估。

        构造两组 logits 均值相同的情况，比较修正前后的池化值。
        """
        g = ExpertGroups.from_sizes((2, 8))   # 故意造一个大小悬殊的分组
        logits = torch.zeros(1, 1, 10)
        pooled = g.pool_to_groups(logits)
        # 两组各自归一后都应为 0，与组大小无关
        assert torch.allclose(pooled, torch.zeros(1, 1, 2), atol=1e-6)

        # 对比：不修正的话 8 元素组的 logsumexp 更大
        raw = torch.stack(
            [torch.logsumexp(logits[..., lo:hi], -1) for lo, hi in g.bounds], -1
        )
        assert raw[0, 0, 1] > raw[0, 0, 0]

    def test_pooling_prefers_correctly_peaked_group(self):
        """正确专家组的一个专家拿到尖峰 logits 时，该组应胜出。"""
        g = DEFAULT_GROUPS
        logits = torch.full((1, 1, 32), -10.0)
        logits[0, 0, 20] = 10.0        # 写作组的一个专家很尖
        pooled = g.pool_to_groups(logits)
        assert int(pooled[0, 0].argmax()) == 2


class TestConfig:
    def test_defaults_are_consistent(self):
        c = Config()
        assert sum(c.group_sizes) == c.num_experts
        assert 0 < c.top_k <= c.num_experts
        assert c.groups.num_experts == c.num_experts

    def test_scaling(self):
        c = Config(lora_rank=16, lora_alpha=16.0)
        assert c.scaling == 1.0
        c2 = Config(lora_rank=8, lora_alpha=16.0)
        assert c2.scaling == 2.0

    def test_group_sizes_must_sum_to_num_experts(self):
        with pytest.raises(ValueError, match="group_sizes"):
            Config(num_experts=32, group_sizes=(8, 8, 8))

    def test_top_k_bounds(self):
        with pytest.raises(ValueError, match="top_k"):
            Config(top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            Config(top_k=33)

    def test_grad_accum_must_be_positive(self):
        with pytest.raises(ValueError, match="grad_accum_steps"):
            Config(grad_accum_steps=0)

    def test_roundtrip(self):
        c = Config(gamma=0.5, top_k=4, seed=7)
        d = c.to_dict()
        assert d["config_version"] == CONFIG_VERSION
        assert Config.from_dict(d).to_dict() == d

    def test_from_dict_ignores_unknown_keys(self):
        """旧 checkpoint 里的多余键不应导致加载失败。"""
        c = Config.from_dict({"gamma": 0.4, "some_legacy_key": 1})
        assert c.gamma == 0.4

    def test_replace_is_immutable(self):
        c = Config()
        c2 = c.replace(gamma=0.9)
        assert c.gamma == 0.3 and c2.gamma == 0.9

    def test_apply_cli_only_overrides_provided(self):
        c = Config()
        import argparse

        ns = argparse.Namespace(gamma=0.7, top_k=None)
        c2 = c.apply_cli(ns)
        assert c2.gamma == 0.7
        assert c2.top_k == c.top_k

    def test_torch_dtype(self):
        import torch

        assert Config(dtype="bfloat16").torch_dtype is torch.bfloat16
        assert Config(dtype="float32").torch_dtype is torch.float32
