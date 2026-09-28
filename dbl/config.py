"""全项目超参的单一事实源。

历史问题：训练脚本与推理脚本各自维护一份 ``TOP_K / GAMMA / LORA_* /
NUM_EXPERTS / GROUP_BOUNDS``，靠注释「必须与…保持一致」人工同步 ——
改一个忘另一个就是静默的数值错位，且极难 debug。

现在两侧都从这里构造 :class:`Config`；checkpoint 也会把它序列化进去，
推理端不再需要手工对齐任何常量。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .groups import DEFAULT_GROUPS, ExpertGroups

CONFIG_VERSION = 2


@dataclass
class Config:
    """模型结构 + 训练超参。

    字段分三类：
      * 结构类（冻结在 checkpoint 里，推理端必须一致）：模型 id、专家数、
        top-k、LoRA rank/alpha、gamma、分组拓扑
      * 训练类：batch、学习率、辅助损失权重
      * 运行类：设备、dtype、路径
    """

    # ---- 结构（写入 checkpoint，推理端据此重建）----
    model_id: str = "Qwen/Qwen3-0.6B"
    num_experts: int = 32
    top_k: int = 8
    lora_rank: int = 16
    lora_alpha: float = 16.0
    gamma: float = 0.3
    group_sizes: tuple[int, ...] = (8, 8, 16)

    # ---- 训练 ----
    micro_batch: int = 4
    grad_accum_steps: int = 4
    max_length: int = 512
    #: 辅助损失权重。直接决定对各层 router 的梯度强度，**不做层间平均**。
    #:
    #: 历史实现把各层 loss 除以 num_layers，但每层 router 都是独立参数，
    #: 对第 i 层求导时系数是 weight/28 —— 辅助信号被稀释 28 倍，等效于
    #: 把 0.1 悄悄降到 0.0036，路由器几乎学不动。
    #:
    #: 注意：调高该权重会让路由更"准"，但在本架构下**文科 PPL 会变差** ——
    #: 详见 README §4 关于设计冲突的说明。默认 0.1 优先保证路由可用。
    router_aux_weight: float = 0.1
    load_balance_weight: float = 0.01
    lr_big_sci: float = 2e-5
    lr_router: float = 3e-4
    lr_experts: float = 5e-4
    weight_decay: float = 0.01
    seed: int = 0
    num_workers: int = 4

    # ---- 运行 ----
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    data_path: str = "dual_contrast_data.jsonl"
    weights_path: str = "dual_big_resurrect_weights.pt"

    # 分组拓扑由 group_sizes 派生，不单独存储
    _groups: ExpertGroups | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.group_sizes = tuple(int(s) for s in self.group_sizes)
        if sum(self.group_sizes) != self.num_experts:
            raise ValueError(
                f"group_sizes 之和 {sum(self.group_sizes)} "
                f"!= num_experts {self.num_experts}"
            )
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k 必须在 1..{self.num_experts}，实际 {self.top_k}"
            )
        if self.lora_rank <= 0:
            raise ValueError("lora_rank 必须为正")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps 必须为正")
        self._groups = ExpertGroups(self._bounds(), self.num_experts)

    def _bounds(self) -> tuple[tuple[int, int], ...]:
        bounds, cur = [], 0
        for s in self.group_sizes:
            bounds.append((cur, cur + s))
            cur += s
        return tuple(bounds)

    @property
    def groups(self) -> ExpertGroups:
        """专家分组拓扑（只读）。"""
        assert self._groups is not None
        return self._groups

    @property
    def scaling(self) -> float:
        """LoRA 缩放系数 alpha / rank。"""
        return self.lora_alpha / self.lora_rank

    @property
    def torch_dtype(self):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

    def replace(self, **kw: Any) -> Config:
        return dataclasses.replace(self, **kw)

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("_groups", None)
        d["group_sizes"] = list(self.group_sizes)
        d["config_version"] = CONFIG_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        d = dict(d)
        d.pop("config_version", None)
        known = {f.name for f in dataclasses.fields(cls)}
        # 容忍历史 checkpoint 里多余的键，避免直接抛错
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def apply_cli(self, args) -> Config:
        """把 argparse 命名空间里显式提供的字段覆盖进来。"""
        overrides = {
            k: v
            for k, v in vars(args).items()
            if k in {f.name for f in dataclasses.fields(self)} and v is not None
        }
        return self.replace(**overrides) if overrides else self


def default_config() -> Config:
    return Config()


__all__ = ["Config", "CONFIG_VERSION", "default_config", "DEFAULT_GROUPS"]
