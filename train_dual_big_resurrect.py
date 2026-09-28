#!/usr/bin/env python
"""训练入口：植入双大核架构并分级学习率训练。

用法::

    python train_dual_big_resurrect.py                       # 默认配置
    python train_dual_big_resurrect.py --config cfg.json     # 读配置文件
    python train_dual_big_resurrect.py --lr-experts 3e-4     # 覆盖单个超参
    python train_dual_big_resurrect.py --resume checkpoints/latest.pt

全部超参定义在 :class:`dbl.config.Config`，与推理端共用同一份。
"""

from __future__ import annotations

import argparse
import logging

from dbl.config import Config
from dbl.runtime import fail_cli
from dbl.train import train


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config json 路径")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None, help="权重输出路径")
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--router-aux-weight", type=float, default=None)
    ap.add_argument("--load-balance-weight", type=float, default=None)
    ap.add_argument("--lr-router", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--delta-dtype", default=None, choices=[None, "int8"],
                    help="Tier-2 存储方式；int8 用相对基座的量化差分，体积约 1/4")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # 配置/设备错误收敛成一行人话，不给调用方一屏 traceback
    try:
        cfg = Config.load(args.config) if args.config else Config()
        if args.out:
            cfg = cfg.replace(weights_path=args.out)
        cfg = cfg.apply_cli(args)
        # 提前验设备：坏参数要在建模（几秒 + 显存）之前就暴露
        cfg = cfg.replace(device=cfg.resolve_device(allow_cpu=False))
    except (RuntimeError, ValueError) as exc:
        fail_cli(exc, "train_dual_big_resurrect.py")
        return

    info = train(
        cfg, resume=args.resume, log_every=args.log_every,
        save_every=args.save_every, delta_dtype=args.delta_dtype,
    )
    print(f"[done] {info}")


if __name__ == "__main__":
    main()
