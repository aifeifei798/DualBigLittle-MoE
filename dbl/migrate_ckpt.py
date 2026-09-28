"""把旧版裸字典 checkpoint 迁移到新格式（带 config 元信息）。

旧格式：``{"layer_0_big_sci": {...}, "layer_0_lora_A": tensor, ...}``
新格式：``{"kind": "dualbiglittle-moe", "config": {...}, "meta": {...}, "state": {...}}``

迁移时把结构超参显式记录进 config，推理端从此不再需要维护第二份常量。
"""

from __future__ import annotations

import argparse

import torch

from dbl.checkpoint import CHECKPOINT_KIND, save_checkpoint
from dbl.config import Config
from dbl.moe import inject_moe

LEGACY_FMT = "legacy-flat-dict"


def detect_format(path: str) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(payload, dict) and payload.get("kind") == CHECKPOINT_KIND:
        return CHECKPOINT_KIND
    return LEGACY_FMT


def migrate(
    src: str, dst: str, cfg: Config, *, device: str = "cpu",
    delta_dtype: str | None = None,
) -> dict:
    """读旧权重 -> 构建模块 -> 按新格式重新保存。

    结构超参必须与当初训练时一致（专家数、top-k、rank、gamma、分组），
    因此这里从 ``cfg`` 读取并写入 checkpoint。
    """
    legacy = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    if legacy.get("kind") == CHECKPOINT_KIND:
        raise SystemExit(f"{src} 已是新格式，无需迁移")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=device
    )
    for p in model.parameters():
        p.requires_grad = False
    mods = inject_moe(model, cfg, mode="train", )

    n_layers = len(model.model.layers)
    keys = {k.split("_")[1] for k in legacy if k.startswith("layer_")}
    found = {int(k) for k in keys}
    if found != set(range(n_layers)):
        raise SystemExit(
            f"权重层数({len(found)}) 与基座({n_layers}) 不一致："
            f"缺失层 {sorted(set(range(n_layers)) - found)}"
        )

    for i, m in enumerate(mods):
        m.big_sci.load_state_dict(legacy[f"layer_{i}_big_sci"])
        m.router_big.load_state_dict(legacy[f"layer_{i}_router_big"])
        m.router_little.load_state_dict(legacy[f"layer_{i}_router_little"])
        m.lora_A.data.copy_(legacy[f"layer_{i}_lora_A"].to(m.lora_A.device,
                                                          m.lora_A.dtype))
        m.lora_B.data.copy_(legacy[f"layer_{i}_lora_B"].to(m.lora_B.device,
                                                          m.lora_B.dtype))
    del legacy

    info = save_checkpoint(
        mods, model, cfg, dst, delta_dtype=delta_dtype,
        extra_meta={"migrated_from": src, "source_format": LEGACY_FMT},
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delta-dtype", default=None, choices=[None, "int8"],
                    help="int8: 存相对基座的量化差分，体积约 1/4")
    ap.add_argument("--num-experts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=float, default=None)
    args = ap.parse_args()

    fmt = detect_format(args.src)
    if fmt == CHECKPOINT_KIND:
        raise SystemExit(f"{args.src} 已是新格式，无需迁移")
    print(f"[migrate] 源格式: {fmt}")

    cfg = Config()
    over = {k: v for k, v in vars(args).items()
            if k in {"num_experts", "top_k", "gamma", "lora_rank",
                     "lora_alpha"} and v is not None}
    if over:
        cfg = cfg.replace(**over)
        print(f"[migrate] 结构超参覆盖: {over}")
    print(f"[migrate] 目标结构: experts={cfg.num_experts} top_k={cfg.top_k} "
          f"rank={cfg.lora_rank} alpha={cfg.lora_alpha} gamma={cfg.gamma} "
          f"groups={cfg.groups.sizes}")

    info = migrate(args.src, args.dst, cfg, device=args.device,
                   delta_dtype=args.delta_dtype)
    print(f"[migrate] 完成: {info}")


if __name__ == "__main__":
    main()
