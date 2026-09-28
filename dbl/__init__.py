"""DualBigLittle-MoE：在消费级显卡上用「大核 + 小核」分层结构做文理双域 MoE。

分层结构（三层物理位置）：

    Tier 1  文科锚核   原版 MLP，完全冻结，常驻显存
    Tier 2  理科孪生核 克隆 MLP，低学习率微调，常驻显存
    Tier 3  微专家池   32 路 rank-r LoRA，置于 pinned 主机内存按需流式搬运

公开接口::

    from dbl import Config, train, evaluate
    cfg = Config()
    train(cfg)
"""

from .checkpoint import (
    apply_checkpoint,
    load_checkpoint,
    load_for_inference,
    save_checkpoint,
)
from .config import CONFIG_VERSION, Config, default_config
from .data import DualContrastDataset
from .groups import DEFAULT_GROUPS, DOMAIN_TO_GROUP, ExpertGroups
from .moe import (
    DualBigLittleMoE,
    InferMoE,
    RouterStats,
    TrainMoE,
    inject_moe,
)

__version__ = "0.2.0"

__all__ = [
    "Config",
    "CONFIG_VERSION",
    "default_config",
    "ExpertGroups",
    "DEFAULT_GROUPS",
    "DOMAIN_TO_GROUP",
    "DualBigLittleMoE",
    "TrainMoE",
    "InferMoE",
    "RouterStats",
    "inject_moe",
    "DualContrastDataset",
    "save_checkpoint",
    "load_checkpoint",
    "apply_checkpoint",
    "load_for_inference",
    "__version__",
]
