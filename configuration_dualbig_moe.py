"""DualBigLittle-MoE 的 Hugging Face 配置定义。

配套 ``modeling_dualbig_moe.py`` 使用，通过 ``trust_remote_code=True`` 加载。

为什么要独立于基座 Qwen3Config
----------------------------
本模型 = Qwen3 骨架 + 每层替换掉的「双大核 + 微专家」FFN。配置必须
**自带全部骨架字段**（hidden_size / num_attention_heads / layer_types /
rope_parameters ...），理由有两条，都不是洁癖：

1. 导出的 ``config.json`` 要能被外部用户的 transformers 独立读懂。若只存
   ``{"base_model": "Qwen/Qwen3-0.6B"}`` 加少量增量字段，模型就必须联网回查
   基座仓库才能构造 —— 这与「一键拉取自包含」直接矛盾。
2. 基座配置在升级中会漂移（字段增删、rope 语义变化）。把当时的取值快照进
   本文件，权重与结构就永久绑定；将来 Qwen3 改了默认值，也不会静默改变
   已发布模型的数值行为。

``Qwen3Model`` 读取的每个字段都在这里显式声明，没有隐式继承。

超参来源
--------
``__init__`` 的默认值取自 ``dbl/config.py``（本项目的单一事实源）在
``dual_big_resurrect_weights.pt`` 训练时的实际取值：专家池 32、Top-k 8、
LoRA rank 16 / alpha 16、gamma 0.3、分组 8/8/16。导出的 ``config.json``
会写入 checkpoint 里的真实值，默认值只用于「未加载权重时的实例化」。
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DualBigMoEConfig(PretrainedConfig):
    r"""DualBigLittle-MoE 的模型配置。

    架构分三层，与 :class:`modeling_dualbig_moe.DualBigMoEMLP` 一一对应：

    ==========  ==================  ==========================
    层          物理位置            内容
    ==========  ==================  ==========================
    Tier 1      显存，冻结          原版 MLP（文科锚核）
    Tier 2      显存，低学习率微调  克隆 MLP（理科孪生核）
    Tier 3      主机 pinned RAM     ``num_experts`` 路 LoRA
    ==========  ==================  ==========================

    Example:

    ```python
    >>> from transformers import AutoConfig
    >>> config = AutoConfig.from_pretrained("your-org/dualbig-qwen3-0.6b", trust_remote_code=True)
    >>> config.num_experts, config.top_k
    (32, 8)
    ```
    """

    model_type = "dualbig_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # ---- 骨架（Qwen3-0.6B 快照，见模块 docstring）----
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    rope_parameters: dict[str, Any] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645

    # ---- 双大核 + 微专家池 ----
    #: Tier-3 每层专家数。896 = 28 层 x 32。
    num_experts: int = 32
    #: 每 token 激活的微专家数。
    #:
    #: **刻意不叫 ``top_k``，也不提供 ``top_k`` 别名。** ``top_k`` 是
    #: ``GenerationConfig`` 的字段名，而 transformers 有两处会按名字撞上它：
    #:
    #:   1. ``PreTrainedModel.__init__`` 调 ``GenerationConfig.from_model_config``，
    #:      把模型 config 里的同名键扫进生成配置 —— 于是「每 token 激活 8 个
    #:      专家」被读成「top-k 采样 k=8」，**静默改变每一次 generate() 的
    #:      输出分布**，且不报任何错；
    #:   2. ``PretrainedConfig._get_generation_parameters`` 用 ``hasattr``
    #:      扫描，所以**连只读 property 形式的别名也会让 save_pretrained 直接
    #:      抛错**。想按惯用读法书写请用 ``config.top_k_experts``。
    top_k_experts: int = 8
    #: 堆叠 LoRA 权重 ``lora_A`` / ``lora_B`` 的形状为 ``(E, rank, hidden)``。
    lora_rank: int = 16
    lora_alpha: float = 16.0
    #: 微专家对大核输出的缩放系数。
    gamma: float = 0.3
    #: 专家分组边界（连续区间，代码 / 数学 / 写作），用于分组级路由监督与遥测展示。
    group_sizes: list[int] | None = None
    #: Tier-3 专家池的存放位置。``"host"`` 走 pinned 主机内存 + PCIe 流式搬运
    #: （已验证的路径）；``"device"`` 常驻显存，跳过搬运，数值完全等价。
    expert_pool_location: str = "host"
    #: 流式搬运的 staging buffer 槽数。2 = 双缓冲，拷贝与计算真正重叠。
    num_staging_buffers: int = 2

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 28,
        layer_types: list[str] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 151643,
        eos_token_id: int | list[int] | None = None,
        num_experts: int = 32,
        top_k_experts: int = 8,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        gamma: float = 0.3,
        group_sizes: list[int] | None = None,
        expert_pool_location: str = "host",
        num_staging_buffers: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # rope 统一走 rope_parameters 字典（transformers 5.x 约定）。
        # 保留 rope_theta 顶层键以兼容仍读旧字段的下游代码。
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(kwargs.pop("rope_theta", 1000000.0)),
            }
        self.rope_parameters = rope_parameters

        self.num_experts = num_experts
        # 兼容手写 config.json 里遗留的 ``top_k``：它曾是本模型的专家 Top-k，
        # 但对 GenerationConfig 而言是采样参数。显式搬过来并丢弃，避免
        # setattr 撞上只读 property 直接抛 AttributeError。
        legacy_top_k = kwargs.pop("top_k", None)
        self.top_k_experts = top_k_experts if legacy_top_k is None else legacy_top_k
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gamma = gamma
        self.expert_pool_location = expert_pool_location
        self.num_staging_buffers = num_staging_buffers

        if group_sizes is None:
            # 默认 8/8/16 = 代码 / 数学 / 写作（与 dbl.groups.DEFAULT_GROUPS 一致）
            group_sizes = self._default_group_sizes(num_experts)
        self.group_sizes = [int(s) for s in group_sizes]

        super().__init__(**kwargs)

        # Qwen3 的 GQA 与滑动窗口约定：KV 头缺省等于 Q 头；未启用滑动窗口时
        # 视为全层 full_attention，缓存与 mask 都依赖这个归一化结果。
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not self.use_sliding_window:
            self.sliding_window = None
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        self._validate()

    @staticmethod
    def _default_group_sizes(num_experts: int) -> list[int]:
        """把 32 路专家按 8/8/16 切成三组；其他专家数按等比缩放。"""
        if num_experts == 32:
            return [8, 8, 16]
        if num_experts <= 0:
            return [num_experts]
        third = max(1, num_experts // 4)
        half = max(1, (num_experts - third) // 2)
        return [third, half, num_experts - third - half]

    def _validate(self) -> None:
        """把结构性错误挡在构造期，而不是留到生成时数值错位。

        这些量一旦与权重不一致就是**静默**的数值错误：模型照常前向，
        只是路由挑了错误的专家。因此必须在构造时炸掉。
        """
        if self.num_experts <= 0:
            raise ValueError(f"num_experts 必须为正，实际 {self.num_experts}")
        if not 0 < self.top_k_experts <= self.num_experts:
            raise ValueError(
                f"top_k_experts 必须落在 1..{self.num_experts}，"
                f"实际 {self.top_k_experts}"
            )
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank 必须为正，实际 {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha 必须为正，实际 {self.lora_alpha}")
        if self.gamma < 0:
            raise ValueError(f"gamma 不能为负，实际 {self.gamma}")
        if self.num_staging_buffers <= 0:
            raise ValueError(
                f"num_staging_buffers 必须为正，实际 {self.num_staging_buffers}"
            )
        if self.expert_pool_location not in ("host", "device"):
            raise ValueError(
                f"expert_pool_location 只能是 'host' 或 'device'，"
                f"实际 {self.expert_pool_location!r}"
            )
        total = sum(self.group_sizes)
        if total != self.num_experts:
            raise ValueError(
                f"group_sizes 之和 {total} 与 num_experts {self.num_experts} 不符"
            )
        if any(s <= 0 for s in self.group_sizes):
            raise ValueError(f"group_sizes 不允许空组：{self.group_sizes}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types 长度 {len(self.layer_types)} 与 "
                f"num_hidden_layers {self.num_hidden_layers} 不符"
            )

    @property
    def lora_scaling(self) -> float:
        """LoRA 缩放系数 ``alpha / rank``。"""
        return self.lora_alpha / self.lora_rank

    @property
    def group_bounds(self) -> list[tuple[int, int]]:
        """把 ``group_sizes`` 展开为连续半开区间，供分组级路由与遥测使用。"""
        bounds, cur = [], 0
        for size in self.group_sizes:
            bounds.append((cur, cur + size))
            cur += size
        return bounds


__all__ = ["DualBigMoEConfig"]
