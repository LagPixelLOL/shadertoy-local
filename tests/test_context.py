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


@pytest.mark.gpu
@pytest.mark.cpu
class TestMultipleContexts:
    def test_each_handle_reports_its_own_device(self, gl_context, software_context):
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
