"""辅助损失权重的回归测试。

这是一次真实事故的防护：``aux_losses`` 曾把 28 层的 loss 除以
``num_layers``。但每层的 router 都是**独立参数**，对第 i 层求导时
系数是 ``weight/28``——辅助信号被稀释 28 倍，等效于把
``router_aux_weight`` 从 0.1 悄悄降到 0.0036。

症状极隐蔽：训练不报错、总 loss 正常下降，只有辅助损失长期停在随机
猜测水平之上（实测 60 步后大核路由准确率 50.0% -> 50.1%，等同随机）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dbl.config import Config
from dbl.moe import RouterStats
from dbl.train import aux_losses

GROUPS = Config(num_experts=8, group_sizes=(2, 2, 4)).groups


class FakeMoE:
    """只需 num_experts 属性的最小替身。"""

    def __init__(self, n):
        self.num_experts = n


def make_stats(n_layers, batch, seqlen, n_experts, groups, seed=0):
    """构造可分数据上的 RouterStats：domain 0 第一维为正，domain 1 为负。"""
    g = torch.Generator().manual_seed(seed)
    stats = []
    for _ in range(n_layers):
        x = torch.randn(batch, seqlen, 16, generator=g)
        x[:, :, 0] += 3.0
        big_logits = torch.stack(
            [torch.zeros(batch, seqlen), x[:, :, 0]], dim=-1
        )
        lit = torch.randn(batch, seqlen, n_experts, generator=g)
        lit[:, :, 0] += 3.0
        probs = torch.softmax(lit, -1)
        topk_w = torch.zeros_like(probs)
        topk_w[:, :, 0] = 1.0
        stats.append(RouterStats(big_logits, lit, probs, topk_w))
    return stats


class TestNoLayerAveraging:
    def test_aux_loss_is_not_divided_by_num_layers(self):
        """核心回归：单层时 aux loss 应等于原始 CE，而非 CE/N。"""
        groups = GROUPS
        n_experts = 8
        stats = make_stats(1, 4, 6, n_experts, groups)
        mods = [FakeMoE(n_experts)]
        mask = torch.ones(4, 6, dtype=torch.bool)
        big_t = torch.zeros(4, dtype=torch.long)
        grp_t = torch.zeros(4, dtype=torch.long)

        big_l, _, _ = aux_losses(mods, stats, big_t, grp_t, mask, groups)

        # 手工算这一层的 CE（标签全 0，第一维 logits 为 3 -> 应高度自信）
        crit = nn.CrossEntropyLoss()
        ref = crit(stats[0].big_logits[mask].float(),
                   torch.zeros(int(mask.sum()), dtype=torch.long))
        assert torch.allclose(big_l, ref, atol=1e-5), (
            f"单层时 aux loss 应等于原始 CE {float(ref):.4f}，"
            f"实际 {float(big_l):.4f}"
        )

    def test_scales_linearly_with_layer_count(self):
        """层数翻倍，aux loss 也应翻倍（求和而非平均）。

        用**完全相同**的 stats 复制成多层，才能验证是求和而非平均。
        """
        n_experts = 8
        mask = torch.ones(4, 6, dtype=torch.bool)
        big_t = torch.zeros(4, dtype=torch.long)
        grp_t = torch.zeros(4, dtype=torch.long)

        one = make_stats(1, 4, 6, n_experts, GROUPS, seed=3)
        two = [one[0], make_stats(1, 4, 6, n_experts, GROUPS, seed=3)[0]]

        l1, _, _ = aux_losses([FakeMoE(n_experts)], one, big_t, grp_t, mask,
                              GROUPS)
        l2, _, _ = aux_losses([FakeMoE(n_experts)] * 2, two, big_t, grp_t, mask,
                              GROUPS)
        assert float(l2) > float(l1), "层数增加时 aux loss 不应变小"
        assert torch.allclose(l2, l1 * 2, atol=1e-5), (
            f"两层应等于单层的 2 倍：{float(l1):.4f} x2 != {float(l2):.4f}"
        )

    def test_dilution_would_have_passed_before(self):
        """反证：若按旧的 /num_layers 归一，测试会失败。"""
        groups = GROUPS
        n_experts, n_layers = 8, 28
        stats = make_stats(n_layers, 4, 6, n_experts, groups)
        mask = torch.ones(4, 6, dtype=torch.bool)
        big_l, _, _ = aux_losses(
            [FakeMoE(n_experts)] * n_layers, stats,
            torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
            mask, groups,
        )
        crit = nn.CrossEntropyLoss()
        per_layer = crit(stats[0].big_logits[mask].float(),
                         torch.zeros(int(mask.sum()), dtype=torch.long))
        old_behavior = float(per_layer)          # 旧实现: /28 后每层梯度系数
        assert float(big_l) > old_behavior * 10, (
            "新实现应远大于单层 CE（未被除以层数）"
        )


class TestAuxWeightSemantics:
    def test_weight_is_configurable_and_linear(self):
        """``router_aux_weight`` 直接决定辅助信号强度，无隐含层数因子。

        验证方式是梯度：对某一层的 router 而言，权重为 w 时其梯度
        应恰为 w 倍（而不是 w/num_layers 倍）。
        """
        n_experts, n_layers, dim = 8, 4, 16
        torch.manual_seed(0)
        routers = [nn.Linear(dim, 2, bias=False) for _ in range(n_layers)]
        little = [nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)]

        n = 32
        x = torch.randn(n, dim)
        big_t = torch.randint(0, 2, (n,),)
        grp_t = torch.randint(0, 3, (n,),)
        mask = torch.ones(n, 1, dtype=torch.bool)

        def grad_at(weight):
            for r in routers + little:
                r.zero_grad()
            stats = []
            for rb, rl in zip(routers, little, strict=True):
                bl = rb(x.unsqueeze(1))
                ll = rl(x.unsqueeze(1))
                p = torch.softmax(ll, -1)
                stats.append(RouterStats(bl, ll, p, p))
            big_l, little_l, _ = aux_losses(
                [FakeMoE(n_experts)] * n_layers, stats, big_t, grp_t, mask,
                GROUPS,
            )
            (weight * (big_l + little_l)).backward()
            return float(routers[0].weight.grad.abs().max())

        g1 = grad_at(1.0)
        g01 = grad_at(0.1)
        assert abs(g01 / g1 - 0.1) < 1e-4, (
            f"梯度应线性于权重，实际比值 {g01 / g1:.4f}（若被 /{n_layers} "
            f"稀释应为 {0.1 / n_layers:.6f}）"
        )


class TestMasking:
    def test_padding_excluded_from_aux(self):
        """padding 位置不得参与辅助损失。"""
        groups = GROUPS
        n_experts = 8
        stats = make_stats(1, 2, 6, n_experts, groups)
        mods = [FakeMoE(n_experts)]
        # 只有前 3 个 token 有效
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, :3] = True
        big_t = torch.zeros(2, dtype=torch.long)
        grp_t = torch.zeros(2, dtype=torch.long)

        big_l, _, _ = aux_losses(mods, stats, big_t, grp_t, mask, groups)
        crit = nn.CrossEntropyLoss()
        gt = big_t.unsqueeze(1).expand(-1, 6)[mask]
        ref = crit(stats[0].big_logits[mask].float(), gt)
        assert torch.allclose(big_l, ref, atol=1e-5)
        assert gt.shape[0] == 6      # 2 batch x 3 token


class TestSeparability:
    def test_aux_can_drive_router_to_perfect_accuracy(self):
        """端到端：只用 aux 优化，可分的 router 必须学到接近 100%。

        这条测试保证辅助损失本身是有效的——若它失效，
        说明问题在别处而非超参。
        """
        groups = GROUPS
        n_experts, n_layers, dim = 8, 2, 16
        g = torch.Generator().manual_seed(0)
        torch.manual_seed(0)

        routers = [nn.Linear(dim, 2, bias=False) for _ in range(n_layers)]
        little = [nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)]

        n = 64
        y = torch.randint(0, 2, (n,), generator=g)
        x = torch.randn(n, dim, generator=g)
        x[:, 0] += (y == 0).float() * 4.0 - (y == 1).float() * 4.0
        yt = torch.randint(0, 3, (n,), generator=g)

        opt = torch.optim.AdamW(
            [p for r in routers + little for p in r.parameters()], lr=1e-2
        )
        crit = nn.CrossEntropyLoss()
        for _ in range(300):
            opt.zero_grad()
            total = 0.0
            for r_big, r_lit in zip(routers, little, strict=True):
                lb = r_big(x)
                ll = r_lit(x)
                total = total + crit(lb, y)
                total = total + crit(groups.pool_to_groups(ll), yt)
            total.backward()
            opt.step()

        acc = (routers[0](x).argmax(-1) == y).float().mean()
        assert acc > 0.95, f"可分数据上 router 只学到 {float(acc):.1%}"
