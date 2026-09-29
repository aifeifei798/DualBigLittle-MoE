"""HF 导出产物的回归测试。

这些断言守的是**静默失败**类问题 —— 它们不会抛异常、不会打日志，只会让
导出后的模型「看起来能跑」但数值已经不是训练验证过的那套。逐条对应一个
已经真实踩过的坑：

* ``test_generation_config_not_polluted_by_expert_top_k``
  ``top_k`` 曾是本模型的专家 Top-k 字段名，而它是 ``GenerationConfig`` 的
  同名字段：``PreTrainedModel.__init__`` 会用 ``from_model_config`` 把它扫进
  生成配置，于是「每 token 激活 8 个专家」变成「top-k 采样 k=8」，**静默
  改变每次 generate 的输出**。改名 ``top_k_experts`` 后需长期守住。
* ``test_host_and_device_pool_agree``
  PCIe 流式路径与零拷贝路径必须逐位一致，否则「换个 pool 位置」就成了
  换一套数值。
* ``test_no_cross_stream_race``
  复刻 commit ``47990f8`` 的竞态回归：同一输入重复前向的 logits 必须完全
  稳定。staging buffer 走 caching allocator 时这里会抖动 1.56e-2。
* ``test_telemetry_survives_meta_device_init``
  ``from_pretrained`` 在 meta device 上构造模块；遥测张量若在 ``__init__``
  里按当时的 device 建好，就会永远停在 meta 上，首次前向直接崩。

无 GPU 时整组跳过（流式路径在 CPU 上退化为单缓冲，覆盖不到竞态）。
"""

from __future__ import annotations

import pytest
import torch

from configuration_dualbig_moe import DualBigMoEConfig
from modeling_dualbig_moe import DualBigMoEForCausalLM, DualBigMoEMLP

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 CUDA 才能覆盖 PCIe 流式搬运路径"
)


def _tiny_config(**overrides) -> DualBigMoEConfig:
    """2 层小模型：结构与 0.6B 同构（双大核 + 堆叠 LoRA），但足够快。"""
    base = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=256,
        num_experts=8,
        top_k_experts=3,
        lora_rank=4,
        lora_alpha=8.0,
        gamma=0.3,
        group_sizes=[2, 2, 4],
        dtype="float32",
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
    )
    base.update(overrides)
    return DualBigMoEConfig(**base)


def _build(pool: str, seed: int = 1234) -> DualBigMoEForCausalLM:
    torch.manual_seed(seed)
    model = DualBigMoEForCausalLM(_tiny_config(expert_pool_location=pool))
    with torch.no_grad():
        # lora_B 零初始化会让所有专家输出恒为 0，那样测不出任何差异
        for layer in model.model.layers:
            layer.mlp.lora_B.normal_(0, 0.05)
    return model.to("cuda").eval()


def test_generation_config_not_polluted_by_expert_top_k():
    """专家 Top-k 绝不能被当成 top-k 采样参数。"""
    config = _tiny_config()
    assert "top_k" not in config.to_dict(), (
        "config 里出现 top_k，会被 GenerationConfig.from_model_config "
        "读成 top-k 采样，静默改变 generate 输出"
    )
    assert config.to_dict()["top_k_experts"] == 3

    model = DualBigMoEForCausalLM(config)
    # 没有显式 generation_config 时，top_k 应保持 GenerationConfig 的
    # 「未设置」状态（None），而不是变成专家 Top-k 的 3。
    assert model.generation_config.top_k in (None, 50), (
        f"top_k 被污染为 {model.generation_config.top_k}"
    )
    assert model.generation_config.top_k != 3


def test_host_and_device_pool_agree():
    """PCIe 流式与零拷贝两条专家池路径必须逐位一致。"""
    host = _build("host")
    device_model = _build("device")
    device_model.load_state_dict(host.state_dict())

    ids = torch.tensor([[1, 5, 9, 3, 7, 2, 11, 4]], device="cuda")
    with torch.no_grad():
        a = host(input_ids=ids).logits.float()
        b = device_model(input_ids=ids).logits.float()

    assert torch.equal(a, b), f"两条 pool 路径不一致: max|diff|={(a - b).abs().max()}"


def test_no_cross_stream_race():
    """重复前向的 logits 必须完全稳定（staging buffer 竞态回归）。"""
    model = _build("host")
    ids = torch.tensor([[1, 5, 9, 3, 7, 2, 11, 4]], device="cuda")

    with torch.no_grad():
        first = model(input_ids=ids).logits.float().clone()
        for _ in range(40):
            again = model(input_ids=ids).logits.float()
            assert torch.equal(first, again), (
                f"重复前向抖动 {float((first - again).abs().max()):.3e}，"
                "staging buffer 很可能又回到了 caching allocator 上"
            )


def test_telemetry_survives_meta_device_init(tmp_path):
    """遥测累加器不能在 meta device 上建好后就再也不动。

    ``from_pretrained`` 走的是「meta 上构造 -> 逐键填权重」。遥测张量不是
    parameter/buffer，``load_state_dict`` 不会碰它们，所以必须在访问时按
    当前 device 惰性重建，否则首次前向就会在 ``index_add_`` 上炸
    ``expected to be in the same device``。
    """
    config = _tiny_config()
    model = DualBigMoEForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)

    reloaded = DualBigMoEForCausalLM.from_pretrained(
        tmp_path, dtype=torch.float32
    ).to("cuda").eval()
    with torch.no_grad():
        reloaded(input_ids=torch.tensor([[1, 2, 3]], device="cuda"))

    tel = reloaded.moe_layers()[0].telemetry
    assert tel["expert_calls"].device.type == "cuda"
    assert int(tel["expert_calls"].sum()) > 0


def test_expert_pool_follows_model_dtype():
    """``from_pretrained`` 后再 ``.to(dtype)``，专家池应自动重建。

    外部用户几乎总是「先 from_pretrained 再 to(cuda)/half」，若专家池的
    dtype 不跟随，就会出现 expert pool 是 bf16 而激活是 fp16 的错配。
    """
    model = DualBigMoEForCausalLM(_tiny_config(dtype="bfloat16")).to("cuda").eval()
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]], device="cuda"))
    first = model.moe_layers()[0]._host_pool["A"].dtype

    model = model.to(torch.float16)
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]], device="cuda"))
    second = model.moe_layers()[0]._host_pool["A"].dtype

    assert first != second, "专家池 dtype 未跟随模型"


def test_lora_b_zero_init_keeps_identity():
    """新初始化时 Tier-3 必须输出 0，不破坏基座行为。

    这是「能从基座起步训练」的前提：B 非零的话，一上来所有专家都会
    往语言底座上加噪声。
    """
    model = DualBigMoEForCausalLM(_tiny_config())
    for mlp in model.moe_layers():
        assert isinstance(mlp, DualBigMoEMLP)
        assert torch.count_nonzero(mlp.lora_B) == 0
        assert torch.count_nonzero(mlp.lora_A) > 0
    # Tier-1 冻结、Tier-2 可训
    for mlp in model.moe_layers():
        assert all(not p.requires_grad for p in mlp.big_arts.parameters())
        assert all(p.requires_grad for p in mlp.big_sci.parameters())


def test_config_rejects_inconsistent_structure():
    """结构与权重不一致是静默的数值错误，必须在构造期就炸。"""
    with pytest.raises(ValueError, match="top_k_experts"):
        DualBigMoEConfig(num_experts=8, top_k_experts=99)
    with pytest.raises(ValueError, match="group_sizes"):
        DualBigMoEConfig(num_experts=32, group_sizes=[8, 8])
    with pytest.raises(ValueError, match="expert_pool_location"):
        DualBigMoEConfig(expert_pool_location="disk")


def test_config_accepts_legacy_top_k_key():
    """旧 config.json 里的 ``top_k`` 应被搬进 top_k_experts 而非丢弃。"""
    config = DualBigMoEConfig(top_k=4)
    assert config.top_k_experts == 4
    assert "top_k" not in config.to_dict()
