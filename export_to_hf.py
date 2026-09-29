#!/usr/bin/env python
"""把双大核权重合并进 Qwen3 基座，导出成标准 Hugging Face 仓库。

产物目录（默认 ``./hf_export``）自带 ``auto_map``，外部用户可以：

.. code-block:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("./hf_export", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./hf_export", trust_remote_code=True)

用法::

    python export_to_hf.py
    python export_to_hf.py --weights checkpoints/v5_auxfix.pt --out hf_export
    python export_to_hf.py --pool-location device   # 跳过 PCIe 流式搬运

命名映射
--------
checkpoint 里是裸字符串键（``layer_0_big_sci``），既没有层级也没有包名，
直接存进 safetensors 会得到一个任何 ``from_pretrained`` 都认不出的文件。
本脚本把它翻译成标准层级路径：

============================  =========================================
checkpoint 键                 导出的 state_dict 键
============================  =========================================
基座 ``mlp.gate_proj``         ``model.layers.N.mlp.big_arts.gate_proj``
``layer_N_big_sci.gate_proj``  ``model.layers.N.mlp.big_sci.gate_proj``
``layer_N_router_big.weight``  ``model.layers.N.mlp.router_big.weight``
``layer_N_router_little.*``    ``model.layers.N.mlp.router_little.weight``
``layer_N_lora_A``             ``model.layers.N.mlp.lora_A``
``layer_N_lora_B``             ``model.layers.N.mlp.lora_B``
============================  =========================================

Tier-1 锚核沿用基座权重、且在 :class:`DualBigMoEMLP` 里被冻结，落到
``big_arts`` 只是为了让「哪个核是冻结的」在权重文件里自解释。

int8 差分
---------
``save_checkpoint(delta_dtype="int8")`` 存的是 Tier-2 相对基座的 per-channel
int8 差分（约 4× 体积收益），必须**依赖逐位一致的基座**才能还原。因此本脚本
会校验 checkpoint 记录的基座指纹，不一致就直接报错 —— 错基座 + 量化差分会
静默还原出错误的权重，比加载失败糟糕得多。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configuration_dualbig_moe import DualBigMoEConfig  # noqa: E402
from modeling_dualbig_moe import DualBigMoEForCausalLM  # noqa: E402

#: 基座仓库里需要原样搬运的分词/生成配置。``chat_template`` 随 Qwen3 的
#: ``tokenizer_config.json`` 一起走，不必单列。
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)

#: Tier-2 差分 fp32 累加时用到的基座 MLP 子模块名。
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class ExportError(RuntimeError):
    """导出过程中的可诊断错误（会被人话化打印，不吐 traceback）。"""


# ----------------------------------------------------------------------
# 权重还原
# ----------------------------------------------------------------------
def _fingerprint(tensor: torch.Tensor) -> str:
    """基座权重指纹。与 ``dbl.checkpoint._fingerprint`` 必须逐位一致。"""
    import hashlib

    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    h.update(flat.numpy().tobytes())
    return h.hexdigest()[:16]


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """读取 dbl checkpoint，返回含 ``Config`` 的 payload。"""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get("kind") != "dualbiglittle-moe":
        raise ExportError(
            f"{path} 不是双大核 checkpoint（kind={payload.get('kind')!r}）。\n"
            "  请用 dbl.migrate_ckpt.migrate() 先迁移旧格式，"
            "或用 --base-config 指定一份 config.json。"
        )
    if "config" not in payload:
        raise ExportError(f"{path} 缺少 config 段，无法确定结构超参")
    return payload


def restore_big_sci(
    layer_idx: int,
    state: dict,
    scales: dict,
    fingerprints: dict,
    base_mlp: dict[str, torch.Tensor],
    delta_dtype: str | None,
) -> dict[str, torch.Tensor]:
    """还原第 ``layer_idx`` 层的 Tier-2 理科大核权重。

    ``delta_dtype`` 为 ``None`` 时 checkpoint 存的就是全量副本，直接取用；
    为 ``"int8"`` 时需要基座权重参与反量化，并逐键校验基座指纹。
    """
    if delta_dtype is None:
        key = f"layer_{layer_idx}_big_sci"
        if key not in state:
            raise ExportError(f"checkpoint 缺少 {key}")
        return {k: v.to(torch.bfloat16) for k, v in state[key].items()}

    restored = {}
    for name in MLP_PROJECTIONS:
        key = f"layer_{layer_idx}_big_sci.{name}.weight"
        if key not in state:
            raise ExportError(f"int8 差分 checkpoint 缺少 {key}")
        base_key = f"{name}.weight"
        if base_key not in base_mlp:
            raise ExportError(
                f"基座第 {layer_idx} 层没有 {base_key}，"
                f"与 checkpoint 的 MLP 结构不兼容"
            )
        base = base_mlp[base_key].to(torch.float32)

        expected = fingerprints.get(key)
        if expected is not None:
            actual = _fingerprint(base_mlp[base_key])
            if actual != expected:
                raise ExportError(
                    f"基座权重与 checkpoint 不匹配：{key}\n"
                    f"  期望指纹 {expected}，实际 {actual}\n"
                    "int8 差分依赖逐位一致的基座。请确认加载的是 checkpoint "
                    f"meta 记录的基座。\n  基座应为："
                    f"{(fingerprints.get('__base__') or '见 checkpoint meta.base_model')}"
                )

        q = state[key].to(torch.float32)
        s = scales[key].to(torch.float32)
        restored[base_key] = (q * s + base).to(torch.bfloat16)
    return restored


# ----------------------------------------------------------------------
# 合并
# ----------------------------------------------------------------------
def build_state_dict(
    base_state: dict[str, torch.Tensor],
    payload: dict,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    """把基座权重与双大核权重合并成标准命名的完整 state_dict。"""
    state = payload["state"]
    meta = payload.get("meta", {})
    scales = payload.get("scales") or {}
    fingerprints = dict(payload.get("fingerprints") or {})
    delta_dtype = meta.get("delta_dtype")

    found = {
        int(k.split("_")[1]) for k in state if k.startswith("layer_") and k.split("_")[1].isdigit()
    }
    missing = set(range(num_layers)) - found
    if missing:
        raise ExportError(
            f"checkpoint 缺少第 {sorted(missing)} 层；"
            f"基座有 {num_layers} 层。权重与基座不匹配，导出后会静默错位。"
        )

    out: dict[str, torch.Tensor] = {}
    for key, tensor in base_state.items():
        # 基座 MLP 的三个投影改挂到 Tier-1 锚核名下
        if ".mlp." in key and key.split(".mlp.")[-1].split(".")[0] in MLP_PROJECTIONS:
            out[key.replace(".mlp.", ".mlp.big_arts.", 1)] = tensor.to(torch.bfloat16)
        else:
            out[key] = tensor.to(torch.bfloat16)

    for i in range(num_layers):
        prefix = f"model.layers.{i}.mlp"

        base_mlp = {
            f"{p}.weight": base_state[f"model.layers.{i}.mlp.{p}.weight"]
            for p in MLP_PROJECTIONS
        }
        for name, tensor in restore_big_sci(
            i, state, scales, fingerprints, base_mlp, delta_dtype
        ).items():
            out[f"{prefix}.big_sci.{name}"] = tensor

        for router in ("router_big", "router_little"):
            key = f"layer_{i}_{router}"
            if key not in state:
                raise ExportError(f"checkpoint 缺少 {key}")
            out[f"{prefix}.{router}.weight"] = state[key]["weight"].to(torch.bfloat16)

        for which in ("lora_A", "lora_B"):
            key = f"layer_{i}_{which}"
            if key not in state:
                raise ExportError(f"checkpoint 缺少 {key}")
            out[f"{prefix}.{which}"] = state[key].to(torch.bfloat16)

    return out


# ----------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------
def build_config(
    base_cfg_dict: dict[str, Any], ckpt_cfg: dict[str, Any], *, pool_location: str
) -> DualBigMoEConfig:
    """以基座配置为骨架、以 checkpoint 为结构超参来源，构造导出配置。

    结构超参（专家数 / Top-k / rank / gamma / 分组）**只认 checkpoint**。
    基座 config 里没有这些字段，训练脚本的 CLI 默认值也不该被拿来猜 ——
    结构与权重不一致是静默的数值错误。
    """
    # dbl 的 Config 里字段就叫 top_k；HF 侧改名 top_k_experts 以避开
    # GenerationConfig 的 top-k 采样参数（见 configuration 的注释）。
    required = ("num_experts", "top_k", "lora_rank", "lora_alpha", "gamma")
    missing = [k for k in required if k not in ckpt_cfg]
    if missing:
        raise ExportError(
            f"checkpoint 的 config 段缺少结构超参 {missing}，拒绝猜测。"
        )

    kwargs = dict(base_cfg_dict)
    for key in ("architectures", "auto_map", "transformers_version", "_name_or_path",
                "model_type", "tokenizer_class", "dtype", "torch_dtype"):
        kwargs.pop(key, None)
    kwargs["num_experts"] = int(ckpt_cfg["num_experts"])
    kwargs["top_k_experts"] = int(ckpt_cfg["top_k"])
    kwargs["lora_rank"] = int(ckpt_cfg["lora_rank"])
    kwargs["lora_alpha"] = float(ckpt_cfg["lora_alpha"])
    kwargs["gamma"] = float(ckpt_cfg["gamma"])
    kwargs["group_sizes"] = list(ckpt_cfg.get("group_sizes") or (8, 8, 16))
    kwargs["expert_pool_location"] = pool_location
    kwargs["dtype"] = "bfloat16"
    kwargs["architectures"] = ["DualBigMoEForCausalLM"]
    kwargs["auto_map"] = {
        "AutoConfig": "configuration_dualbig_moe.DualBigMoEConfig",
        "AutoModelForCausalLM": "modeling_dualbig_moe.DualBigMoEForCausalLM",
    }
    kwargs["base_model_name_or_path"] = ckpt_cfg.get("model_id", "Qwen/Qwen3-0.6B")

    try:
        return DualBigMoEConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"基座配置字段与 DualBigMoEConfig 不兼容：{exc}") from exc


def copy_tokenizer(base_model: str, out: Path) -> list[str]:
    """把分词器与生成配置搬到导出目录。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(out)

    gen_src = _hub_file(base_model, "generation_config.json")
    if gen_src is not None:
        shutil.copy2(gen_src, out / "generation_config.json")

    # save_pretrained 已覆盖主要文件；下面补齐它可能不写的老式文件。
    hub_dir = Path(_snapshot_dir(base_model))
    copied = []
    for name in TOKENIZER_FILES:
        src = hub_dir / name
        if src.exists() and not (out / name).exists():
            shutil.copy2(src, out / name)
            copied.append(name)
    return copied


def _snapshot_dir(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["*.json", "*.txt"])


def _hub_file(model_id: str, filename: str) -> str | None:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(model_id, filename)
    except Exception:
        return None


def verify(model: DualBigMoEForCausalLM, out: Path, device: str) -> dict[str, Any]:
    """导出后立刻重载并前向一次，确保产物真的能被 ``from_pretrained`` 用起来。"""
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(
        out, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()

    with torch.no_grad():
        ids = torch.tensor([[128000, 790, 6864, 315]], device=device)
        logits = reloaded(input_ids=ids).logits

    report = {
        "reloaded_class": type(reloaded).__name__,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_std": float(logits.float().std()),
    }
    if not report["logits_finite"]:
        raise ExportError(
            "重载后前向产生 NaN/Inf。常见原因：专家池未随权重装载，"
            "或 gamma / top_k_experts 与权重结构不一致。"
        )
    del reloaded
    return report


def export(
    weights_path: str,
    out_dir: str,
    *,
    base_model: str | None = None,
    pool_location: str = "host",
    max_shard_size: str = "2GB",
    device: str = "cuda:0",
    skip_verify: bool = False,
) -> dict[str, Any]:
    """执行一次完整导出，返回摘要字典。"""
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint(weights_path)
    ckpt_cfg = payload["config"]
    meta = payload.get("meta", {})
    base_model = base_model or ckpt_cfg.get("model_id") or meta.get("base_model")
    if not base_model:
        raise ExportError("无法确定基座模型：请显式传 --base-model")

    # ---- 基座权重与配置 ----
    from transformers import AutoConfig, AutoModelForCausalLM

    base_cfg = AutoConfig.from_pretrained(base_model)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16
    )
    num_layers = base_model_obj.config.num_hidden_layers
    base_state = base_model_obj.state_dict()
    del base_model_obj

    ckpt_layers = meta.get("num_layers")
    if ckpt_layers is not None and ckpt_layers != num_layers:
        raise ExportError(
            f"checkpoint 记录 {ckpt_layers} 层，基座 {base_model} 是 {num_layers} 层。"
        )

    merged = build_state_dict(base_state, payload, num_layers)
    config = build_config(base_cfg.to_dict(), ckpt_cfg, pool_location=pool_location)

    # ---- 装载到本仓模型类，再由 save_pretrained 落盘 ----
    #
    # 刻意走 save_pretrained 而非手写 safetensors：tied embedding 的去重、
    # 分片索引、metadata、generation_config 全由 transformers 处理。手写这几件事
    # 正是「权重能加载但 lm_head 是随机初始化」这类事故的来源。
    model = DualBigMoEForCausalLM(config)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    missing = [k for k in missing if "rotary_emb" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise ExportError(
            f"权重与模型结构不匹配。\n  缺失: {missing[:8]}\n"
            f"  多余: {unexpected[:8]}"
        )
    model = model.to(torch.bfloat16)

    model.save_pretrained(out, safe_serialization=True, max_shard_size=max_shard_size)

    # ---- 代码 + 分词器 + 卡片 ----
    here = Path(__file__).resolve().parent
    for name in ("configuration_dualbig_moe.py", "modeling_dualbig_moe.py"):
        shutil.copy2(here / name, out / name)
    extra_tok = copy_tokenizer(base_model, out)
    shutil.copy2(here / "example_usage.py", out / "example_usage.py")
    (out / "README.md").write_text(
        render_readme(config, meta, weights_path, base_model), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "output": str(out),
        "base_model": base_model,
        "num_layers": num_layers,
        "num_experts": config.num_experts,
        "top_k_experts": config.top_k_experts,
        "lora_rank": config.lora_rank,
        "gamma": config.gamma,
        "group_sizes": config.group_sizes,
        "expert_pool_location": config.expert_pool_location,
        "delta_dtype": meta.get("delta_dtype"),
        "total_experts": config.num_experts * num_layers,
        "extra_tokenizer_files": extra_tok,
    }

    if not skip_verify:
        summary["verify"] = verify(model, out, device)

    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return summary


# ----------------------------------------------------------------------
# 附带文件
# ----------------------------------------------------------------------


def render_readme(
    config: DualBigMoEConfig, meta: dict, weights_path: str, base_model: str
) -> str:
    quant = meta.get("quant_rel_error")
    quant_line = (
        f"- Tier-2 差分 int8 量化相对误差：`{quant}`\n" if quant is not None else ""
    )
    return f"""---
library_name: transformers
tags:
- dualbig-moe
- qwen3
- moe
- custom_code
license: apache-2.0
base_model: {base_model}
---

# DualBigLittle-MoE · Qwen3-0.6B

**双大核 + 微专家**架构，在消费级显卡上同时保住文科底座与理科能力。

| 层 | 位置 | 内容 |
| :-- | :-- | :-- |
| Tier 1 文科锚核 | 显存（冻结） | 原版 MLP，保证不退化 |
| Tier 2 理科孪生核 | 显存（微调） | 克隆 MLP，代码 / 数学 / 逻辑 |
| Tier 3 微专家池 | {'主机 pinned RAM（PCIe 流式）' if config.expert_pool_location == 'host' else '显存（零拷贝）'} | {config.num_experts * config.num_hidden_layers} 个 rank-{config.lora_rank} LoRA，每层选 Top-{config.top_k_experts} |

## 结构超参

| 项 | 值 |
| :-- | --: |
| 隐藏维 / 层数 | {config.hidden_size} / {config.num_hidden_layers} |
| 注意力头 (Q/KV) | {config.num_attention_heads} / {config.num_key_value_heads} |
| 微专家数 / 每层 | {config.num_experts} |
| Top-k | {config.top_k_experts} |
| LoRA rank / alpha | {config.lora_rank} / {config.lora_alpha} |
| gamma | {config.gamma} |
| 专家分组（代码/数学/写作） | {config.group_sizes} |
| 专家池位置 | {config.expert_pool_location} |

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "."  # 或 "your-org/dualbig-qwen3-0.6b"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=torch.bfloat16
).to("cuda").eval()

messages = [{{"role": "user", "content": "解释一下快速排序"}}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(text, return_tensors="pt").to("cuda"), max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
```

完整示例（含流式输出与路由遥测）见 `example_usage.py`。

## 路由遥测

```python
model.reset_routing_stats()
model.generate(...)          # 或直接 forward 一段文本
model.routing_report()       # 双大核能量分配 + 专家/分组调用分布
```

## 权重来源

- 基座：`{base_model}`
- 训练权重：`{weights_path}`
- 保存时间：{meta.get('saved_at', '未知')}
{quant_line}
## 说明

本模型使用 `trust_remote_code=True` 加载 `modeling_dualbig_moe.py`。
该文件在 Qwen3 骨架上替换每层 FFN，并实现 Tier-3 的 CUDA 异步双缓冲搬运
（常驻 staging buffer + ping-pong 事件，不使用 `record_stream` ——
原因见 modeling 文件的模块 docstring）。

只想跑推理、不需要 PCIe 搬运时，可在 config.json 里设
`"expert_pool_location": "device"`，数值结果不变。
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", default="dual_big_resurrect_weights.pt")
    ap.add_argument("--out", default="hf_export")
    ap.add_argument("--base-model", default=None,
                    help="缺省从 checkpoint 的 config.model_id 读取")
    ap.add_argument("--pool-location", default="host", choices=("host", "device"))
    ap.add_argument("--max-shard-size", default="2GB")
    ap.add_argument("--device", default="cuda:0", help="导出后自检使用的设备")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    try:
        summary = export(
            args.weights,
            args.out,
            base_model=args.base_model,
            pool_location=args.pool_location,
            max_shard_size=args.max_shard_size,
            device=args.device,
            skip_verify=args.skip_verify,
        )
    except ExportError as exc:
        print(f"export_to_hf.py: 错误：{exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n导出完成 -> {summary['output']}")
    print("自检：AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)")


if __name__ == "__main__":
    main()
