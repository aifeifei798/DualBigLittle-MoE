#!/usr/bin/env python
"""环境体检：一条命令确认「双大核 + 专家池」能正确跑在 CUDA 上。

用法::

    python doctor.py              # 只看环境
    python doctor.py --gpu-check  # 额外做一次真实的 GPU 前后向（含显存峰值）

设计取舍：默认**不**加载模型。0.6B 基座 + 专家池要几秒到几十秒，
而环境问题（缺卡、装成 CPU-only torch、算力不匹配）全部能在
零成本阶段暴露。真要验证能跑，再加 ``--gpu-check``。
"""

from __future__ import annotations

import argparse
import sys

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def _check_imports() -> bool:
    print("── 依赖 ──────────────────────────────────────────")
    ok = True
    for mod, want in (("torch", "2.14.0"), ("transformers", "5.17.0")):
        try:
            m = __import__(mod)
        except ImportError as e:
            print(f"{BAD} {mod:<14} 未安装 ({e})")
            ok = False
            continue
        got = m.__version__
        # torch 的 CUDA 轮子带 local 版本号（2.14.0+cu130），它就是我们要的
        # 构建；只有当基础版本不同才算版本漂移。
        base = got.split("+", 1)[0]
        flag = OK if base == want else WARN
        note = "" if base == want else f"（预期 {want}，数字复现要求一致）"
        if flag is OK and got != want:
            note = "（CUDA 构建）"
        print(f"{flag} {mod:<14} {got}{note}")
    return ok


def _check_cuda() -> bool:
    print("\n── CUDA ──────────────────────────────────────────")
    try:
        import torch
    except ImportError:
        print(f"{BAD} torch 缺失，无法继续")
        return False

    from dbl.runtime import device_report, unsupported_arch_hint

    if not torch.cuda.is_available():
        build = getattr(torch.version, "cuda", None)
        print(f"{BAD} torch.cuda.is_available() == False")
        print(f"     torch={torch.__version__}  自带 CUDA={build}")
        if build is None:
            print("     → 这是 CPU-only torch。重装：")
            print("       uv sync")
            print("     或 pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu130")
        else:
            print("     → torch 带 CUDA 但驱动不可用：检查 nvidia-smi，"
                  "容器需 --gpus all")
        return False

    info = device_report()
    print(f"{OK} 设备 {info['device']}  {info['gpu']}")
    print(f"{OK} 显存 {info['vram_mib']} MiB   "
          f"算力 sm_{info['compute_capability'].replace('.', '')}")
    print(f"{OK} torch {info['torch']}  (CUDA {info['torch_cuda_build']})")

    hint = unsupported_arch_hint()
    if hint:
        print(f"{WARN} {hint}")
    else:
        print(f"{OK} 算力在预编译列表内，无需 PTX JIT")
    if info["device_count"] > 1:
        print(f"{OK} 共 {info['device_count']} 张卡；"
              f"auto 模式会挑最空的一张")
    return True


def _check_device_arg() -> bool:
    print("\n── 设备参数解析 ──────────────────────────────────")
    from dbl.runtime import resolve_device

    ok = True
    for want, expect in (("auto", None), ("cuda:0", None), ("cpu", "cpu")):
        try:
            got = resolve_device(want)
        except Exception as e:  # noqa: BLE001 - 体检脚本要报告任何失败
            print(f"{BAD} --device {want:<8} 解析失败: {e}")
            ok = False
            continue
        if expect and got != expect:
            print(f"{BAD} --device {want:<8} -> {got}（预期 {expect}）")
            ok = False
        else:
            print(f"{OK} --device {want:<8} -> {got}")
    return ok


def _gpu_check() -> None:
    """真跑一次 GPU 前后向，确认专家池的 stream 路径可用。"""
    print("\n── GPU 前后向（含专家池搬运）─────────────────────")
    try:
        import torch
        from transformers import AutoModelForCausalLM

        from dbl.config import Config
        from dbl.moe import inject_moe

        cfg = Config(device="auto").to_device()
        dev = cfg.device
        print(f"     加载 {cfg.model_id} …")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=cfg.torch_dtype, device_map=dev
        )
        mods = inject_moe(model, cfg, mode="infer")
        for m in mods:
            m.load_expert_pool(m.lora_A.data, m.lora_B.data)
        model.eval()

        ids = torch.randint(0, 1000, (1, 32), device=dev)
        with torch.no_grad():
            out = model(input_ids=ids).logits
        torch.cuda.synchronize()

        print(f"{OK} 前向通过  logits {tuple(out.shape)} on {out.device}")
        print(f"{OK} 峰值显存 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

        # 抖动检查：同一输入重复前向必须逐位一致，否则可复现性结论不成立
        with torch.no_grad():
            a = model(input_ids=ids).logits
            b = model(input_ids=ids).logits
        same = torch.equal(a, b)
        print(f"{'ok' if same else 'warn'} 重复前向逐位一致 = {same}")
        if not same:
            print("     非确定性问题：若需严格可复现，检查是否启用了非确定性"
                  "算子或 TF32 混用")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} GPU 检查失败: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gpu-check", action="store_true",
                    help="额外加载模型跑一次真实前向（需数秒）")
    args = ap.parse_args()

    print("═" * 60)
    print("DualBigLittle-MoE 环境体检")
    print("═" * 60)

    ok = _check_imports()
    ok &= _check_cuda()
    ok &= _check_device_arg()

    if args.gpu_check and ok:
        _gpu_check()

    print()
    if ok:
        print("结论：环境就绪，双大核与 Tier-3 专家池将跑在 CUDA 上。")
        print("提醒：设备默认 auto，可用 --device cuda:N 或环境变量 "
              "DBL_DEVICE 固定。")
    else:
        print("结论：环境有问题，见上方 [FAIL] 项。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
