"""CUDA 运行时检查与设备解析。

历史问题：设备字符串散落在各处硬编码（``"cuda:0"`` / ``"cpu"``），既没有
校验也没有统一入口。后果是三类静默故障：

  1. 在没有显卡的机器上跑到一半才炸，报错信息指向 stream/event 而不是根因；
  2. 指定的 ``cuda:1`` 不存在，错误来自 ``from_pretrained`` 的深层调用栈；
  3. 显卡算力不在 torch 编译的 arch 列表里，退化成 PTX JIT，首个 kernel
     编译要几十秒，表现为「卡住了」而不是「不兼容」。

本模块把设备解析集中到一个入口 :func:`resolve_device`，在**构造任何张量
之前**就把上述问题变成一句能照着修的报错，并在启动时打印
:func:`device_report` 供确认。
"""

from __future__ import annotations

import os
from typing import Any

#: 需要时自动探测的设备占位符。
AUTO = "auto"


def cuda_available() -> bool:
    """torch 能否真正驱动 CUDA（不只看 ``torch.version.cuda``）。"""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 是硬依赖
        return False
    return bool(torch.cuda.is_available())


def device_count() -> int:
    import torch

    return torch.cuda.device_count()


def best_device() -> str:
    """挑一张「当前最空」的卡；并列时取编号最小的。"""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = None
    for i in range(torch.cuda.device_count()):
        try:
            free_i, _total = torch.cuda.mem_get_info(i)
        except Exception:  # pragma: no cover - 驱动异常时按 0 处理
            free_i = 0
        if free is None or free_i > free:
            free, idx = free_i, i
    return f"cuda:{idx}"


def resolve_device(requested: str | None = None, *, allow_cpu: bool = True) -> str:
    """把用户给的设备字符串解析成一个真正可用的设备。

    Args:
        requested: ``None`` / ``"auto"`` 表示自动探测；``"cpu"`` 表示强制
            CPU（测试与无显卡调试用）；``"cuda"`` / ``"cuda:1"`` 表示指定卡。
        allow_cpu: 显式 CUDA 请求在**没有可用 GPU** 时是否允许退回 CPU。
            默认 ``True``，因为 pytest 与纯 CPU 单测必须能跑；
            训练/评估入口应传 ``False``，让缺卡立刻暴露而不是退化成
            「慢 50 倍但结果看起来正常」。

    Returns:
        归一化后的设备字符串。

    Raises:
        RuntimeError: 显式要求 CUDA 但环境不满足，且 ``allow_cpu=False``。
    """
    req = (requested or AUTO).strip().lower()

    if req == "cpu":
        return "cpu"

    if req in (AUTO, "cuda"):
        if cuda_available():
            return best_device()
        if req == AUTO and allow_cpu:
            return "cpu"
        raise RuntimeError(_no_cuda_message(requested))

    if not req.startswith("cuda:"):
        raise ValueError(
            f"无法识别的设备 {requested!r}；"
            "支持 'auto' / 'cpu' / 'cuda' / 'cuda:N'"
        )

    # 显式 cuda:N —— 校验索引存在且 torch 真的能驱动
    suffix = req.split(":", 1)[1]
    if not suffix.isdigit():
        raise ValueError(f"设备 {requested!r} 的下标必须是整数")
    idx = int(suffix)
    if not cuda_available():
        if allow_cpu:
            return "cpu"
        raise RuntimeError(_no_cuda_message(requested))
    if idx >= device_count():
        raise RuntimeError(
            f"请求 {requested!r}，但只检测到 {device_count()} 张 GPU。"
            f"可用设备：{[f'cuda:{i}' for i in range(device_count())]}"
        )
    return f"cuda:{idx}"


def _no_cuda_message(requested: str | None) -> str:
    import torch

    build = getattr(torch.version, "cuda", None)
    return (
        f"要求 CUDA 设备（{requested!r}）但当前环境不可用。\n"
        f"  torch          : {torch.__version__}\n"
        f"  torch.version.cuda: {build}\n"
        f"  torch.cuda.is_available(): {torch.cuda.is_available()}\n"
        "排查：\n"
        "  1) 确认机器有 NVIDIA 卡且 `nvidia-smi` 正常；\n"
        "  2) 装 CPU-only torch 会导致这里为 False——需要 CUDA 版：\n"
        "       uv pip install torch --index-url "
        "https://download.pytorch.org/whl/cu130\n"
        "  3) 容器环境需要 `--gpus all` 传入设备。"
    )


def unsupported_arch_hint() -> str | None:
    """若当前卡的算力不在 torch 的编译列表里，返回一条提示，否则 ``None``。

    命中不代表跑不了——会退回 PTX JIT 现场编译，只是首个 kernel 明显变慢，
    所以值得提前说一句。
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(0)
    built = set(torch.cuda.get_arch_list())
    if not built:
        return None
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm in built:
        return None
    # torch 2.9+ 会给 arch 列表加后缀（如 sm_90a / sm_100f）
    if any(b.split("_")[0] == sm or b.startswith(sm) for b in built):
        return None
    return (
        f"当前 GPU 算力 {sm} 不在 torch 预编译列表 {sorted(built)} 中，"
        "将以 PTX JIT 方式运行，首个 kernel 编译耗时较长。"
    )


def device_report() -> dict[str, Any]:
    """汇总当前 CUDA 环境，供启动日志 / 评估报告使用。"""
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available(),
        "device": resolve_device(AUTO),
    }
    if info["cuda_available"]:
        i = 0
        info.update(
            gpu=torch.cuda.get_device_name(i),
            compute_capability=".".join(map(str, torch.cuda.get_device_capability(i))),
            arch_list=sorted(torch.cuda.get_arch_list()),
            device_count=device_count(),
            vram_mib=round(
                torch.cuda.get_device_properties(i).total_memory / 2**20
            ),
        )
        hint = unsupported_arch_hint()
        if hint:
            info["arch_warning"] = hint
    return info


def format_report(info: dict[str, Any] | None = None) -> str:
    """把 :func:`device_report` 渲染成多行文本（启动时打印）。"""
    info = info or device_report()
    if not info["cuda_available"]:
        return (
            "CUDA 不可用 —— 本项目的大核/专家池设计依赖 GPU，"
            "在 CPU 上会退化且慢很多。\n"
            f"  torch={info['torch']} cuda_build={info['torch_cuda_build']}"
        )
    lines = [
        f"  设备      : {info['device']}  ({info['gpu']})",
        f"  算力      : sm_{info['compute_capability'].replace('.', '')}"
        f"   显存 {info['vram_mib']} MiB",
        f"  torch     : {info['torch']}  (CUDA {info['torch_cuda_build']})",
        f"  arch 列表 : {', '.join(info['arch_list'])}",
    ]
    if "arch_warning" in info:
        lines.append(f"  ⚠ {info['arch_warning']}")
    return "\n".join(lines)


def log_device_banner(logger, *, title: str = "运行环境") -> None:
    """把设备信息打进 logging，便于在训练日志里留痕。"""
    try:
        info = device_report()
    except Exception as exc:  # pragma: no cover - 探测失败不应阻断主流程
        logger.warning("设备探测失败：%s", exc)
        return
    logger.info("%s\n%s", title, format_report(info))


def default_device_from_env() -> str:
    """读取 ``DBL_DEVICE`` 环境变量；未设置则自动探测。"""
    return resolve_device(os.environ.get("DBL_DEVICE") or AUTO)


def fail_cli(exc: BaseException, prog: str) -> None:
    """把配置类错误变成一行人话，而不是一屏 traceback。

    设备写错、没装 CUDA、config 字段非法这类问题，调用方几乎不可能从
    栈底部的 ``RuntimeError`` 里读出「该改哪个参数」。这里统一收敛成
    ``prog: 错误原因`` 并以非零码退出。
    """
    import sys

    print(f"{prog}: 错误：{exc}", file=sys.stderr)
    hint = getattr(exc, "hint", None)
    if hint:
        print(f"  提示：{hint}", file=sys.stderr)
    sys.exit(2)


__all__ = [
    "AUTO",
    "best_device",
    "cuda_available",
    "default_device_from_env",
    "device_count",
    "device_report",
    "fail_cli",
    "format_report",
    "log_device_banner",
    "resolve_device",
    "unsupported_arch_hint",
]
