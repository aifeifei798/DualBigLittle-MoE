"""专家分组拓扑 —— 全项目唯一定义。

历史问题：分组边界曾在 `train_dual_big_resurrect.py`、`chat_dual_big_resurrect.py`、
`prepare_dual_data.py` 三处各写一份，靠注释「必须与…保持一致」人工同步。
本模块是唯一定义处，其余代码一律从这里取。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# 领域名 -> 分组下标。数据集标注与本表共用。
DOMAIN_TO_GROUP: dict[str, int] = {"Code": 0, "Math": 1, "Arts": 2}

GROUP_LABELS: tuple[str, ...] = ("代码", "数学", "写作")


@dataclass(frozen=True)
class ExpertGroups:
    """专家分组的连续区间划分。

    `bounds` 为半开区间 [lo, hi) 的序列，必须连续覆盖 ``[0, num_experts)``。
    各组大小可以不等 —— 分组级路由损失里的 ``log(size)`` 修正正是为此存在。
    """

    bounds: tuple[tuple[int, int], ...]
    num_experts: int

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("bounds 不能为空")
        if self.bounds[0][0] != 0:
            raise ValueError(f"分组必须从 0 开始，实际为 {self.bounds[0][0]}")
        for (_lo, hi), (nlo, _) in zip(self.bounds, self.bounds[1:], strict=False):
            if hi != nlo:
                raise ValueError(f"分组区间不连续：{hi} != {nlo}")
        if self.bounds[-1][1] != self.num_experts:
            raise ValueError(
                f"分组边界必须覆盖 0..{self.num_experts}，"
                f"实际结束于 {self.bounds[-1][1]}"
            )
        for lo, hi in self.bounds:
            if lo >= hi:
                raise ValueError(f"空分组：[{lo}, {hi})")

    @classmethod
    def from_sizes(cls, sizes: Sequence[int]) -> ExpertGroups:
        """按各组专家数构造，自动生成连续边界。"""
        bounds: list[tuple[int, int]] = []
        cur = 0
        for s in sizes:
            bounds.append((cur, cur + s))
            cur += s
        return cls(tuple(bounds), cur)

    @property
    def num_groups(self) -> int:
        return len(self.bounds)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(hi - lo for lo, hi in self.bounds)

    def group_of_expert(self, expert: int) -> int:
        for g, (lo, hi) in enumerate(self.bounds):
            if lo <= expert < hi:
                return g
        raise ValueError(f"专家下标 {expert} 越界")

    def group_of_domain(self, domain: str) -> int:
        try:
            return DOMAIN_TO_GROUP[domain]
        except KeyError:
            raise ValueError(
                f"未知领域 {domain!r}，可选：{sorted(DOMAIN_TO_GROUP)}"
            ) from None

    def log_size_correction(self) -> tuple[float, ...]:
        """各组的 ``log(size)``，用于分组级路由池化的规模归一。

        各组大小不等时，裸 ``logsumexp`` 会系统性偏向大组，减去它才公平。
        """
        return tuple(math.log(hi - lo) for lo, hi in self.bounds)

    def pool_to_groups(self, expert_logits):
        """把专家级 logits 池化为分组级 logits。

        在最后一维按区间做 ``logsumexp`` 并减去 ``log(size)``。
        """
        import torch

        return torch.stack(
            [
                torch.logsumexp(expert_logits[..., lo:hi], dim=-1)
                - math.log(hi - lo)
                for lo, hi in self.bounds
            ],
            dim=-1,
        )

    def expert_share(self, counts) -> list[float]:
        """由各专家调用次数算出各组占比（用于遥测展示）。"""
        total = float(sum(counts))
        if total <= 0:
            return [0.0] * self.num_groups
        return [
            sum(counts[lo:hi]) / total for lo, hi in self.bounds
        ]

    def labels(self) -> tuple[str, ...]:
        return GROUP_LABELS[: self.num_groups]


#: 默认拓扑：32 个专家分成 8 / 8 / 16，对应 代码 / 数学 / 写作
DEFAULT_GROUPS = ExpertGroups.from_sizes((8, 8, 16))
