"""梯度累积缩放的单元测试。

这是一次真实事故的回归防护：缩放曾写成 ``1/min(remainder, N)``，
使组内权重变成 1, 1/2, 1/3, 1/4（合计 2.083 而非 1.0），
把每个累积组第一个 micro-batch 的梯度放大 4 倍。
症状极隐蔽 —— 训练不报错、loss 也在降，但路由器辅助损失长期停在
随机猜测水平之上，领域路由完全学不出来。
"""

from __future__ import annotations

import torch

from dbl.train import grad_accum_scales


class TestScales:
    def test_full_group_sums_to_one(self):
        """一个完整累积组内，所有 micro-batch 的缩放之和应为 1.0。"""
        total_batches, n = 16, 4
        s = grad_accum_scales(total_batches, n)
        assert len(s) == total_batches
        for start in range(0, total_batches, n):
            group = s[start : start + n]
            assert abs(sum(group) - 1.0) < 1e-9, f"组 {start} 权重和 {sum(group)}"

    def test_full_group_is_uniform(self):
        """完整组内每个 micro-batch 权重必须相同。"""
        s = grad_accum_scales(16, 4)
        first = s[0:4]
        assert all(abs(x - 0.25) < 1e-9 for x in first), first

    def test_no_batch_dominates(self):
        """任一 micro-batch 的权重不得超过平均值的 1.01 倍。"""
        s = grad_accum_scales(20, 4)
        avg = 1 / 4
        assert max(s) <= avg * 1.01, f"最大权重 {max(s)} 远高于均值 {avg}"

    def test_partial_final_group_rescaled(self):
        """末尾不足 N 个时，按实际个数缩放（而不是沿用 1/N 低估）。"""
        s = grad_accum_scales(18, 4)      # 18 = 4*4 + 2
        tail = s[16:18]
        assert len(tail) == 2
        assert abs(sum(tail) - 1.0) < 1e-9
        # 应当是 1/2 而不是 1/4
        assert all(abs(x - 0.5) < 1e-9 for x in tail), tail

    def test_exact_multiple(self):
        s = grad_accum_scales(8, 4)
        assert all(abs(x - 0.25) < 1e-9 for x in s)
        assert abs(sum(s) - 2.0) < 1e-9    # 两个完整组

    def test_fewer_batches_than_group(self):
        s = grad_accum_scales(3, 4)
        assert abs(sum(s) - 1.0) < 1e-9
        assert all(abs(x - 1 / 3) < 1e-9 for x in s)

    def test_single_micro_batch(self):
        s = grad_accum_scales(1, 4)
        assert s == [1.0]

    def test_total_weight_is_number_of_steps(self):
        """全程累计权重应等于优化步数。"""
        for total, n in [(16, 4), (18, 4), (7, 3), (100, 4), (5, 8)]:
            s = grad_accum_scales(total, n)
            assert abs(sum(s) - round(total / n if total % n == 0
                                     else total // n + 1)) < 1e-9


class TestEffectiveGradient:
    def test_uniform_group_matches_single_large_batch(self):
        """把 4 个等分 micro-batch 累积起来，应等价于一个整体反向。

        这是缩放正确性的端到端检验：权重不均时两者必然不等。
        """
        torch.manual_seed(0)
        total, n = 8, 4
        scales = grad_accum_scales(total, n)

        # 模拟：4 个 micro-batch 的梯度
        micro_grads = [torch.randn(5) for _ in range(n)]
        accumulated = sum(g * scales[i] for i, g in enumerate(micro_grads))
        whole = sum(micro_grads) / n
        assert torch.allclose(accumulated, whole, atol=1e-6)

    def test_wrong_scaling_would_fail(self):
        """反证：旧的错误缩放确实会导致不等。"""
        micro_grads = [torch.randn(5) for _ in range(4)]
        bad = sum(g * (1.0 / ((i % 4) + 1)) for i, g in enumerate(micro_grads))
        whole = sum(micro_grads) / 4
        assert not torch.allclose(bad, whole, atol=1e-3)
