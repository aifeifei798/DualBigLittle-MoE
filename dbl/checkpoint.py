"""checkpoint 的保存与加载。

历史问题：权重是裸字符串键字典（``layer_0_big_sci``），不带任何超参信息。
推理端必须自己维护一份 ``TOP_K / GAMMA / LORA_* / GROUP_BOUNDS`` 常量，
与训练脚本各写一份 —— 靠注释「必须与…保持一致」人工同步。

现在 checkpoint 自带 :class:`Config` 与元信息，推理端从文件反序列化重建，
常量漂移在结构上被消除。

关于体积：Tier-2 理科大核是基座 MLP 的克隆，只保存全量副本有大量冗余
（实测 Δ 仅占权重的 0.11%~0.21%）。但 bf16 存差分**不省空间** ——
差分并不稀疏，28 层仍是 504 MiB。要真正缩小必须量化：
``delta_dtype="int8"`` 用 per-channel scale，约 126 MiB（4× 收益），
且 0.2% 的相对扰动远低于 int8 分辨率。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .config import CONFIG_VERSION, Config
from .moe import DualBigLittleMoE, InferMoE

CHECKPOINT_KIND = "dualbiglittle-moe"


def _module_key(index: int, name: str) -> str:
    return f"layer_{index}_{name}"


def _fingerprint(t: torch.Tensor) -> str:
    """张量的内容指纹，用于校验基座权重是否与保存时一致。"""
    import hashlib

    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    flat = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
    h.update(flat.numpy().tobytes())
    return h.hexdigest()[:16]


def _delta_quantize(
    trained: torch.Tensor, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-channel int8 量化差分，返回 ``(量化值, scale)``。

    scale 按输出通道（最后一维）计算，保住每通道的幅度。
    全程在 CPU 上做，避免训练时模块在 GPU 而参考权重在 CPU 造成设备错配。
    """
    t = trained.detach().to("cpu", torch.float32)
    b = base.detach().to("cpu", torch.float32)
    d = t - b
    scale = d.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp_min(1e-12)
    q = torch.round(d / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


def _delta_dequantize(
    q: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    out = q.float() * scale + base.detach().to("cpu", torch.float32)
    return out.to(base.dtype)


def _reference_mlp(mod: DualBigLittleMoE):
    """Tier-2 差分的参考权重来源。

    注意：``inject_moe`` 执行后 ``layer.mlp`` 已经被替换成 MoE 模块本身，
    因此基座参考权重必须取 ``mod.big_arts``（Tier-1，保持冻结，
    与原始 MLP 逐位相同），而不是 ``layer``。
    """
    return mod.big_arts


def save_checkpoint(
    modules: list[DualBigLittleMoE],
    base_model=None,
    cfg: Config | None = None,
    path: str | Path | None = None,
    *,
    delta_dtype: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """保存 checkpoint。

    ``delta_dtype``:
        ``None``  -> Tier-2 存全量副本（体积大但完全精确，默认）
        ``"int8"`` -> Tier-2 存相对基座的 int8 差分（约 4× 体积收益）
    """
    if cfg is None or path is None:
        raise ValueError("save_checkpoint 需要显式提供 cfg 与 path")
    state: dict[str, Any] = {}
    quantized = 0
    scales: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    max_quant_rel_err = 0.0

    for i, mod in enumerate(modules):
        if delta_dtype == "int8":
            ref = dict(_reference_mlp(mod).named_parameters())
            for k, v in mod.big_sci.state_dict().items():
                if k not in ref:
                    raise KeyError(
                        f"层 {i} 的 big_sci 键 {k!r} 在基座 MLP 中不存在"
                    )
                base_t = ref[k].detach()
                fingerprints[_module_key(i, f"big_sci.{k}")] = _fingerprint(base_t)
                q, s = _delta_quantize(v, base_t)
                state[_module_key(i, f"big_sci.{k}")] = q
                scales[_module_key(i, f"big_sci.{k}")] = s
                # 此时 trained 与 base 都在手，可直接测真实量化误差
                restored = _delta_dequantize(q, s, base_t)
                v_cpu = v.detach().to("cpu", torch.float32)
                denom = float(v_cpu.abs().max())
                if denom > 0:
                    rel = float((restored.float() - v_cpu).abs().max()) / denom
                    max_quant_rel_err = max(max_quant_rel_err, rel)
            quantized += 1
        else:
            state[_module_key(i, "big_sci")] = {
                k: v.detach().cpu() for k, v in mod.big_sci.state_dict().items()
            }
        state[_module_key(i, "router_big")] = {
            k: v.detach().cpu() for k, v in mod.router_big.state_dict().items()
        }
        state[_module_key(i, "router_little")] = {
            k: v.detach().cpu() for k, v in mod.router_little.state_dict().items()
        }
        state[_module_key(i, "lora_A")] = mod.lora_A.detach().cpu()
        state[_module_key(i, "lora_B")] = mod.lora_B.detach().cpu()

    payload = {
        "kind": CHECKPOINT_KIND,
        "config_version": CONFIG_VERSION,
        "config": cfg.to_dict(),
        "meta": {
            "base_model": cfg.model_id,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_layers": len(modules),
            "delta_dtype": delta_dtype,
            "num_quantized_layers": quantized,
            "quant_rel_error": (
                round(max_quant_rel_err, 8) if delta_dtype == "int8" else None
            ),
            **(extra_meta or {}),
        },
        "state": state,
    }
    if scales:
        payload["scales"] = scales
    if fingerprints:
        payload["fingerprints"] = fingerprints

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_mib = path.stat().st_size / 2**20
    return {
        "path": str(path),
        "size_mib": round(size_mib, 1),
        "delta_dtype": delta_dtype,
        "num_layers": len(modules),
        "quant_rel_error": payload["meta"].get("quant_rel_error"),
    }


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
    mmap: bool = True,
) -> dict[str, Any]:
    """加载 checkpoint，返回含解析后 ``Config`` 的 payload。

    同时兼容旧的裸字典格式（无 ``kind`` 字段），此时 Config 需由调用方提供。
    """
    payload = torch.load(
        path, map_location=map_location, weights_only=True, mmap=mmap
    )
    if isinstance(payload, dict) and payload.get("kind") == CHECKPOINT_KIND:
        payload["cfg"] = Config.from_dict(payload["config"])
        return payload
    # 旧格式：裸 layer_* 键
    return {
        "kind": "legacy",
        "state": payload,
        "config": None,
        "cfg": None,
        "meta": {"format": "legacy-flat-dict"},
    }


def apply_checkpoint(
    modules: list[DualBigLittleMoE],
    payload: dict[str, Any],
    base_model=None,
) -> dict[str, Any]:
    """把 checkpoint 里的权重装进已注入的模块。"""
    state = payload["state"]
    meta = payload.get("meta", {})
    delta_dtype = meta.get("delta_dtype")
    scales = payload.get("scales") or {}
    fingerprints = payload.get("fingerprints") or {}
    loaded = 0

    for i, mod in enumerate(modules):
        if delta_dtype == "int8":
            ref = dict(_reference_mlp(mod).named_parameters())
            new_state = {}
            for k, base_t in ref.items():
                key = _module_key(i, f"big_sci.{k}")
                if key not in state:
                    raise KeyError(f"checkpoint 缺少 {key}")
                # 差分是相对基座定义的：基座不一致会静默还原出错误权重，
                # 因此必须逐键校验后者的指纹。
                expected = fingerprints.get(key)
                if expected is not None:
                    actual = _fingerprint(base_t)
                    if actual != expected:
                        raise ValueError(
                            f"基座权重与 checkpoint 不匹配：{key}\n"
                            f"  期望指纹 {expected}，实际 {actual}\n"
                            "int8 差分依赖逐位一致的基座。请确认加载的是"
                            f" checkpoint meta 记录的基座："
                            f"{meta.get('base_model')}"
                        )
                restored = _delta_dequantize(state[key], scales[key], base_t)
                new_state[k] = restored
            mod.big_sci.load_state_dict(new_state)
        else:
            mod.big_sci.load_state_dict(state[_module_key(i, "big_sci")])
        loaded += 1
        mod.router_big.load_state_dict(state[_module_key(i, "router_big")])
        mod.router_little.load_state_dict(state[_module_key(i, "router_little")])
        mod.lora_A.data.copy_(state[_module_key(i, "lora_A")].to(mod.lora_A.device))
        mod.lora_B.data.copy_(state[_module_key(i, "lora_B")].to(mod.lora_B.device))

    return {
        "format": payload.get("kind"),
        "delta_dtype": delta_dtype,
        # 量化误差在**保存时**测得并存入 meta（加载时已无 trained 权重可比）
        "quant_rel_error": meta.get("quant_rel_error"),
        "loaded_layers": loaded,
    }


def load_for_inference(path: str | Path, base_model, cfg: Config | None = None):
    """推理端便捷入口：装载权重并把专家池 pin 到主机内存。

    超参优先取 checkpoint 自带的 ``config``（消除推理端手工同步常量的
    需要）；若为旧格式且未提供 cfg，则报错而不是猜。
    """
    payload = load_checkpoint(path)
    if payload.get("cfg") is None and cfg is None:
        raise ValueError(
            f"{path} 是旧格式 checkpoint，不含超参；"
            "请显式传入 cfg 以避免训练/推理配置不一致"
        )
    mods = [layer.mlp for layer in base_model.model.layers]
    info = apply_checkpoint(mods, payload)
    for mod in mods:
        if isinstance(mod, InferMoE):
            mod.load_expert_pool(mod.lora_A.data, mod.lora_B.data)
    return payload, info


__all__ = [
    "save_checkpoint", "load_checkpoint", "apply_checkpoint",
    "load_for_inference", "CHECKPOINT_KIND",
]
