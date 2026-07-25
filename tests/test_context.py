"""Device selection and context creation.

The selection logic is pure Python and deserves direct tests: it is the only
thing standing between a user and a silent ~100x slowdown from rendering on a
CPU rasterizer by accident.
"""

from __future__ import annotations

import pytest

from shadertoy_local.context import (
    ContextError,
    DeviceInfo,
    enumerate_devices,
    select_device,
)

SOFTWARE = ("EGL_MESA_device_software", "EGL_EXT_device_drm_render_node")
NVIDIA = ("EGL_NV_device_cuda", "EGL_EXT_device_drm")
GENERIC = ("EGL_EXT_device_drm",)


def _devices(*specs: tuple[int, tuple[str, ...]]) -> list[DeviceInfo]:
    return [DeviceInfo(index=i, extensions=exts) for i, exts in specs]


class TestDeviceInfo:
    def test_software_detection(self):
        (dev,) = _devices((0, SOFTWARE))
        assert dev.is_software
        assert not dev.is_nvidia

    def test_nvidia_detection(self):
        (dev,) = _devices((0, NVIDIA))
        assert dev.is_nvidia
        assert not dev.is_software

    def test_label_prefers_renderer_string(self):
        dev = DeviceInfo(index=0, extensions=NVIDIA, renderer="RTX 9999")
        assert dev.label == "RTX 9999"

    def test_label_falls_back_to_kind(self):
        assert _devices((0, SOFTWARE))[0].label == "software rasterizer"
        assert _devices((0, NVIDIA))[0].label == "NVIDIA device"
        assert _devices((0, ()))[0].label == "unknown device"

    def test_serialisable(self):
        import json

        payload = _devices((0, SOFTWARE))[0].to_dict()
        assert json.loads(json.dumps(payload))["software"] is True


class TestSelectDevice:
    def test_prefers_hardware_even_when_software_is_first(self):
        """llvmpipe is frequently enumerated first; picking it would be a
        silent, enormous slowdown."""
        devices = _devices((0, SOFTWARE), (1, NVIDIA))
        assert select_device(devices).index == 1

    def test_refuses_software_by_default(self):
        devices = _devices((0, SOFTWARE))
        assert select_device(devices) is None

    def test_allow_software_permits_fallback(self):
        devices = _devices((0, SOFTWARE))
        assert select_device(devices, allow_software=True).index == 0

    def test_allow_software_does_not_override_available_hardware(self):
        """--allow-software only *permits* CPU as a fallback; it must never
        take precedence over a working GPU."""
        devices = _devices((0, SOFTWARE), (1, NVIDIA))
        assert select_device(devices, allow_software=True).index == 1

    def test_explicit_index_wins(self):
        devices = _devices((0, SOFTWARE), (1, NVIDIA))
        assert select_device(devices, requested=0).index == 0

    def test_explicit_software_index_needs_no_allow_flag(self):
        """Naming a device is explicit intent, so it bypasses the guard."""
        devices = _devices((0, SOFTWARE), (1, NVIDIA))
        chosen = select_device(devices, requested=0, allow_software=False)
        assert chosen.index == 0 and chosen.is_software

    def test_unknown_index_lists_what_exists(self):
        devices = _devices((0, SOFTWARE), (1, NVIDIA))
        with pytest.raises(ContextError, match=r"does not exist \(available: 0, 1\)"):
            select_device(devices, requested=7)

    def test_nvidia_preferred_among_hardware(self):
        devices = _devices((0, GENERIC), (1, NVIDIA))
        assert select_device(devices).index == 1

    def test_lowest_index_among_equal_hardware(self):
        devices = _devices((0, GENERIC), (1, GENERIC))
        assert select_device(devices).index == 0

    def test_empty_list(self):
        assert select_device([]) is None
        assert select_device([], requested=0) is None


@pytest.mark.gpu
class TestRealEnumeration:
    def test_enumeration_returns_devices(self):
        devices = enumerate_devices()
        assert devices, "expected at least one EGL device"
        assert all(isinstance(d.index, int) for d in devices)

    def test_indices_are_contiguous_from_zero(self):
        devices = enumerate_devices()
        assert [d.index for d in devices] == list(range(len(devices)))

    def test_software_flag_agrees_with_the_renderer_string(self):
        """Cross-check capability detection against the driver's self-report.

        `is_software` keys off EGL_MESA_device_software, which is how devices are
        classified before any context exists. If that ever disagreed with the
        actual GL_RENDERER, auto-selection could quietly land on a CPU
        rasterizer.
        """
        from shadertoy_local.context import ContextError, create_context

        known_software = ("llvmpipe", "softpipe", "swrast", "swiftshader")
        for dev in enumerate_devices():
            try:
                handle = create_context(
                    device_index=dev.index, allow_software=True
                )
            except ContextError:
                continue
            try:
                renderer = handle.ctx.info["GL_RENDERER"].lower()
            finally:
                handle.release()
            looks_software = any(name in renderer for name in known_software)
            assert dev.is_software == looks_software, (
                f"device {dev.index} ({renderer}) flagged "
                f"is_software={dev.is_software}"
            )


@pytest.mark.gpu
@pytest.mark.cpu
class TestMultipleContexts:
    def test_each_handle_reports_its_own_device(
        self, gl_context, software_context, auto_device_only
    ):
        """Regression: moderngl caches Context.info on first access and the query
        reads whichever context is current, so a lazily-read handle could report
        a different device's renderer -- and claim software=False for llvmpipe."""
        assert gl_context.to_dict()["software"] is False
        assert software_context.to_dict()["software"] is True
        assert "llvmpipe" in software_context.to_dict()["gl_renderer"].lower()
        assert "llvmpipe" not in gl_context.to_dict()["gl_renderer"].lower()

    def test_both_contexts_remain_usable(self, gl_context, software_context):
        import numpy as np

        def clear_and_read(handle, value):
            ctx = handle.ctx
            texture = ctx.texture((4, 4), 4, dtype="f4")
            fbo = ctx.framebuffer(color_attachments=[texture])
            try:
                fbo.use()
                fbo.clear(value, 0.0, 0.0, 1.0)
                ctx.finish()
                raw = fbo.read(components=4, dtype="f4")
                return float(np.frombuffer(raw, dtype=np.float32)[0])
            finally:
                fbo.release()
                texture.release()

        # Interleaved use must not bleed between contexts.
        assert clear_and_read(gl_context, 0.25) == pytest.approx(0.25)
        assert clear_and_read(software_context, 0.75) == pytest.approx(0.75)
        assert clear_and_read(gl_context, 0.5) == pytest.approx(0.5)


class TestMultiGpuRanking:
    """Ordering on hosts with several GPUs. Enumerating a device is not a
    promise it can be bound, so ranking must produce a usable fallback order."""

    def test_two_nvidia_gpus_prefer_lowest_index(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, NVIDIA), (1, NVIDIA))
        assert [d.index for d in rank_devices(devices)] == [0, 1]

    def test_nvidia_outranks_other_hardware_regardless_of_order(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, GENERIC), (1, NVIDIA), (2, GENERIC))
        assert [d.index for d in rank_devices(devices)] == [1, 0, 2]

    def test_software_is_excluded_from_ranking_by_default(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, SOFTWARE), (1, NVIDIA), (2, GENERIC))
        assert [d.index for d in rank_devices(devices)] == [1, 2]

    def test_software_ranks_last_when_allowed(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, SOFTWARE), (1, GENERIC), (2, NVIDIA))
        ranked = rank_devices(devices, allow_software=True)
        assert [d.index for d in ranked] == [2, 1, 0]
        assert ranked[-1].is_software, "a CPU rasterizer must never outrank a GPU"

    def test_multiple_software_devices_keep_index_order(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, SOFTWARE), (1, SOFTWARE))
        assert [d.index for d in rank_devices(devices, allow_software=True)] == [0, 1]

    def test_explicit_request_yields_exactly_one_candidate(self):
        """An explicit --device must never silently fall back to other hardware:
        a measurement taken on a substituted GPU would be meaningless."""
        from shadertoy_local.context import rank_devices

        devices = _devices((0, NVIDIA), (1, NVIDIA), (2, SOFTWARE))
        assert [d.index for d in rank_devices(devices, requested=1)] == [1]

    def test_all_hardware_only_no_software_available(self):
        from shadertoy_local.context import rank_devices

        devices = _devices((0, NVIDIA), (1, GENERIC))
        assert len(rank_devices(devices, allow_software=True)) == 2


class TestDeviceFallback:
    """create_context must try the next candidate when one cannot be bound."""

    @pytest.fixture(autouse=True)
    def _hermetic_env(self, monkeypatch):
        """These tests drive selection directly with a stubbed driver, so the
        ambient SHADERTOY_* steering used to pin the suite to one device must not
        leak in and change what is attempted."""
        for name in (
            "SHADERTOY_DEVICE",
            "SHADERTOY_ALLOW_SOFTWARE",
            "SHADERTOY_BACKEND",
        ):
            monkeypatch.delenv(name, raising=False)

    def _fake_devices(self, monkeypatch, devices):
        import shadertoy_local.context as ctxmod

        monkeypatch.setattr(ctxmod, "enumerate_devices", lambda: devices)

    def _fake_driver(self, monkeypatch, failing: set[int]):
        """Stub moderngl.create_context, failing for the given device indices."""
        import moderngl

        class FakeInfo(dict):
            pass

        class FakeCtx:
            def __init__(self, index):
                self.device_index = index
                self.version_code = 460
                self.info = FakeInfo({"GL_RENDERER": f"fake-{index}"})

            def release(self):
                pass

        def fake_create(**kwargs):
            index = kwargs.get("device_index")
            if index in failing:
                raise RuntimeError(f"cannot bind device {index}")
            return FakeCtx(index)

        monkeypatch.setattr(moderngl, "create_context", fake_create)

    def test_falls_through_to_the_second_gpu(self, monkeypatch):
        from shadertoy_local.context import create_context

        self._fake_devices(monkeypatch, _devices((0, NVIDIA), (1, NVIDIA)))
        self._fake_driver(monkeypatch, failing={0})
        handle = create_context()
        assert handle.device is not None and handle.device.index == 1

    def test_reports_every_failure_when_all_devices_fail(self, monkeypatch):
        from shadertoy_local.context import ContextError, create_context

        self._fake_devices(monkeypatch, _devices((0, NVIDIA), (1, GENERIC)))
        self._fake_driver(monkeypatch, failing={0, 1})
        with pytest.raises(ContextError) as excinfo:
            create_context()
        message = str(excinfo.value)
        assert "device 0" in message and "device 1" in message

    def test_explicit_device_does_not_fall_back(self, monkeypatch):
        from shadertoy_local.context import ContextError, create_context

        self._fake_devices(monkeypatch, _devices((0, NVIDIA), (1, NVIDIA)))
        self._fake_driver(monkeypatch, failing={0})
        with pytest.raises(ContextError):
            create_context(device_index=0)

    def test_does_not_fall_back_to_software_unless_allowed(self, monkeypatch):
        from shadertoy_local.context import ContextError, create_context

        self._fake_devices(monkeypatch, _devices((0, NVIDIA), (1, SOFTWARE)))
        self._fake_driver(monkeypatch, failing={0})
        with pytest.raises(ContextError):
            create_context()

    def test_falls_back_to_software_when_allowed(self, monkeypatch):
        from shadertoy_local.context import create_context

        self._fake_devices(monkeypatch, _devices((0, NVIDIA), (1, SOFTWARE)))
        self._fake_driver(monkeypatch, failing={0})
        handle = create_context(allow_software=True)
        assert handle.device is not None and handle.device.index == 1
