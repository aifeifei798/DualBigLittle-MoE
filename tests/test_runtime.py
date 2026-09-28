"""设备解析的单元测试（无需 GPU）。

这些测试通过 monkeypatch :func:`torch.cuda.is_available` 等来模拟
「有卡 / 无卡 / 卡数变化」，因此在 CI 的 CPU 机器上同样能跑，
且能覆盖真实机器上很难构造的分支（比如请求不存在的 cuda:7）。
"""

from __future__ import annotations

import pytest
import torch

from dbl import runtime
from dbl.config import Config


@pytest.fixture
def gpu(monkeypatch):
    """模拟一台 n 张卡的机器。"""

    def _make(n: int = 1, free: int = 8 << 30):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: n > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info",
            lambda i=None: (free - i * (1 << 20), 1 << 34),
        )
        return n

    return _make


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


class TestResolveDevice:
    def test_cpu_is_always_honored(self, no_gpu):
        assert runtime.resolve_device("cpu") == "cpu"
        assert runtime.resolve_device("CPU") == "cpu"

    def test_auto_prefers_gpu_when_present(self, gpu):
        gpu(1)
        assert runtime.resolve_device("auto") == "cuda:0"

    def test_none_means_auto(self, gpu):
        gpu(1)
        assert runtime.resolve_device(None) == "cuda:0"

    def test_auto_falls_back_to_cpu(self, no_gpu):
        assert runtime.resolve_device("auto", allow_cpu=True) == "cpu"

    def test_explicit_cuda_survives_within_range(self, gpu):
        gpu(4)
        for i in range(4):
            assert runtime.resolve_device(f"cuda:{i}") == f"cuda:{i}"

    def test_explicit_cuda_raises_past_device_count(self, gpu):
        gpu(2)
        with pytest.raises(RuntimeError, match="只检测到 2 张 GPU"):
            runtime.resolve_device("cuda:7", allow_cpu=False)

    def test_bad_index_raises_even_with_allow_cpu(self, gpu):
        """下标写错必须报错，不能因为 allow_cpu 就悄悄退回 CPU。

        allow_cpu 表达的是「这台机器根本没有 GPU」，而不是「你把卡号写错了」。
        后者静默降级会让人以为在 GPU 上跑、实际拿到 CPU 的慢结果。
        """
        gpu(2)
        with pytest.raises(RuntimeError, match="只检测到 2 张 GPU"):
            runtime.resolve_device("cuda:9", allow_cpu=True)

    def test_no_gpu_and_disallowed_raises_actionable(self, no_gpu):
        with pytest.raises(RuntimeError) as ei:
            runtime.resolve_device("auto", allow_cpu=False)
        msg = str(ei.value)
        # 报错必须能照着修：点明是 CPU-only torch、并给出安装命令
        assert "download.pytorch.org" in msg
        assert "is_available" in msg

    def test_allow_cpu_false_blocks_silent_fallback(self, no_gpu):
        with pytest.raises(RuntimeError):
            runtime.resolve_device("cuda:0", allow_cpu=False)

    @pytest.mark.parametrize("bad", ["gpu:0", "cuda:", "cuda:x", "tpu", "CUDA 0"])
    def test_rejects_garbage(self, bad, gpu):
        gpu(1)
        with pytest.raises(ValueError):
            runtime.resolve_device(bad, allow_cpu=False)

    def test_bare_cuda_picks_a_real_device(self, gpu):
        gpu(3)
        assert runtime.resolve_device("cuda") in {"cuda:0", "cuda:1", "cuda:2"}

    def test_picks_least_loaded_card(self, gpu):
        """「auto」应挑最空的一张卡，而不是永远 cuda:0。"""
        gpu(3)
        # monkeypatch 后 free 随 i 递减 -> cuda:0 最空
        assert runtime.best_device() == "cuda:0"


class TestArchHint:
    def test_no_warning_when_arch_supported(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (12, 0))
        monkeypatch.setattr(
            torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90", "sm_120"]
        )
        assert runtime.unsupported_arch_hint() is None

    def test_warns_when_arch_missing(self, monkeypatch):
        """不在列表里仍能跑（PTX JIT），但值得提前说一句。"""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_120"])
        hint = runtime.unsupported_arch_hint()
        assert hint is not None and "sm_75" in hint

    def test_tolerates_arch_suffixes(self, monkeypatch):
        """torch 2.9+ 会给 arch 加后缀（sm_90a / sm_100f），不能误报。"""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (9, 0))
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90a", "sm_120f"])
        assert runtime.unsupported_arch_hint() is None

    def test_no_hint_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert runtime.unsupported_arch_hint() is None


class TestDeviceReport:
    def test_report_keys_on_gpu(self, gpu):
        gpu(1)
        info = runtime.device_report()
        assert info["cuda_available"] is True
        assert "sm_" in "sm_" + info["compute_capability"].replace(".", "")
        assert isinstance(info["arch_list"], list)
        assert runtime.format_report(info)

    def test_report_without_cuda_does_not_crash(self, no_gpu):
        """旧实现直接 get_device_name(0)，无卡机器上这里会 AttributeError。"""
        info = runtime.device_report()
        assert info["cuda_available"] is False
        assert info["device"] == "cpu"
        assert "CUDA 不可用" in runtime.format_report(info)

    def test_banner_never_raises(self, monkeypatch, no_gpu):
        """探测失败只应告警，不应阻断训练主流程。"""
        import logging

        monkeypatch.setattr(
            runtime, "device_report", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        runtime.log_device_banner(logging.getLogger("t"), title="x")


class TestConfigIntegration:
    def test_default_is_auto_not_hardcoded_card(self):
        assert Config().device == "auto"

    def test_to_device_materializes_concrete_card(self, gpu):
        gpu(2)
        cfg = Config()
        assert cfg.to_device().device in {"cuda:0", "cuda:1"}

    def test_to_device_is_idempotent(self, gpu):
        gpu(1)
        once = Config().to_device()
        assert once.to_device().device == once.device

    def test_resolve_device_can_veto_cpu(self, no_gpu):
        with pytest.raises(RuntimeError):
            Config(device="auto").resolve_device(allow_cpu=False)

    def test_env_var_overrides_auto(self, gpu, monkeypatch):
        """DBL_DEVICE 只需设一次，对所有入口生效。"""
        gpu(4)
        monkeypatch.setenv("DBL_DEVICE", "cuda:2")
        assert Config(device="auto").resolve_device() == "cuda:2"

    def test_env_var_ignored_when_device_explicit(self, gpu, monkeypatch):
        """显式 --device / config 里的 device 优先于环境变量。"""
        gpu(4)
        monkeypatch.setenv("DBL_DEVICE", "cuda:3")
        assert Config(device="cuda:1").resolve_device() == "cuda:1"

    def test_env_var_bad_value_is_reported(self, gpu, monkeypatch):
        """环境变量写错也要给出人话报错，而不是静默忽略。"""
        gpu(1)
        monkeypatch.setenv("DBL_DEVICE", "cuda:8")
        with pytest.raises(RuntimeError, match="只检测到 1 张 GPU"):
            Config(device="auto").resolve_device(allow_cpu=False)

    def test_env_var_cpu_still_vetoed_when_gpu_required(self, no_gpu, monkeypatch):
        """DBL_DEVICE=cpu 不能绕过 allow_cpu=False 的强制上卡。"""
        monkeypatch.setenv("DBL_DEVICE", "cpu")
        assert Config(device="auto").resolve_device(allow_cpu=True) == "cpu"

    def test_no_env_var_falls_back_to_probe(self, gpu, monkeypatch):
        gpu(1)
        monkeypatch.delenv("DBL_DEVICE", raising=False)
        assert Config(device="auto").resolve_device() == "cuda:0"

    def test_config_roundtrip_keeps_device(self, gpu):
        gpu(1)
        cfg = Config(device="cuda:0")
        assert Config.from_dict(cfg.to_dict()).device == "cuda:0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要真实 GPU")
    def test_real_gpu_is_usable(self):
        """真机冒烟：解析出的设备必须真能建张量、算对、并回读。"""
        dev = runtime.resolve_device("auto", allow_cpu=False)
        n = 4
        t = torch.ones(n, n, device=dev)
        # (n x n 全 1) @ (n x n 全 1) = 每个元素 n，共 n*n 个 -> 总和 n^3
        expected = float(n**3)
        assert (t @ t).sum().item() == pytest.approx(expected, rel=1e-3)
        # 结果必须真的落在那张卡上，而不是静默回落到 CPU
        assert (t @ t).device.type == "cuda"
        assert runtime.device_report()["cuda_available"] is True

    def test_build_model_resolves_auto_device(self, monkeypatch):
        """回归：默认 ``Config()`` 的 device 是 "auto"。

        历史上 build_model 不解析设备，直接把 "auto" 传给 from_pretrained /
        inject_moe，于是任何自建 Config 的调用方（诊断脚本、实验代码）
        都会在 ``.to("auto")`` 上炸：
            RuntimeError: Expected one of cpu, cuda, ... at start of device
        这里用 stub 替掉 from_pretrained / inject_moe，只验证「传进去的
        是不是具体设备串」，不必真的下载模型。
        """
        import dbl.train as train_mod

        seen: dict[str, str] = {}

        class _StubModel:
            def parameters(self):
                return iter(())

        monkeypatch.setattr(
            train_mod.AutoTokenizer, "from_pretrained",
            classmethod(lambda cls, *a, **k: type(
                "T", (), {"pad_token": None, "eos_token": None})()),
        )
        monkeypatch.setattr(
            train_mod.AutoModelForCausalLM, "from_pretrained",
            classmethod(lambda cls, *a, **k: (seen.update(
                device_map=k.get("device_map")) or _StubModel())),
        )
        monkeypatch.setattr(
            train_mod, "inject_moe", lambda model, cfg, **k: seen.update(
                inject_device=cfg.device) or [],
        )

        train_mod.build_model(Config(device="auto"))

        assert seen["device_map"] not in (None, "auto"), (
            f"from_pretrained 收到了未解析的设备串 {seen['device_map']!r}"
        )
        assert seen["inject_device"] not in (None, "auto")
        assert torch.device(seen["inject_device"]).type in {"cpu", "cuda"}
