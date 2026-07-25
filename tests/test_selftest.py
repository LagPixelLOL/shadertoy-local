"""Per-device runtime verification.

The point of these tests is that the check can actually *fail*. A verification
that always passes is worse than none, since it manufactures false confidence, so
each failure mode is induced deliberately and checked for the right stage.
"""

from __future__ import annotations

import numpy as np
import pytest

import shadertoy_local.selftest as selftest
from shadertoy_local.selftest import (
    RuntimeCheck,
    check_all_devices,
    check_device,
    expected_image,
)


@pytest.fixture
def restore_fragment():
    """Let a test corrupt the check shader, then put it back."""
    original = selftest._FRAGMENT
    yield lambda text: setattr(selftest, "_FRAGMENT", text)
    selftest._FRAGMENT = original


class TestExpectedImage:
    def test_shape_and_channels(self):
        image = expected_image(8)
        assert image.shape == (8, 8, 4)

    def test_samples_pixel_centres(self):
        """gl_FragCoord lands at pixel centres, so the first pixel is 0.5/N."""
        image = expected_image(8)
        assert image[0, 0, 0] == pytest.approx(0.5 / 8 * selftest._SCALE)
        assert image[0, 0, 1] == pytest.approx(0.5 / 8 * selftest._SCALE)

    def test_exceeds_one(self):
        """Values above 1.0 are deliberate: an 8-bit target would clamp them and
        the check would pass on a pipeline that silently loses range."""
        assert expected_image(8).max() > 1.0

    def test_constant_channels(self):
        image = expected_image(4)
        assert np.allclose(image[..., 2], selftest._ARRAY_X)
        assert np.allclose(image[..., 3], 1.0)


@pytest.mark.gpu
class TestPassingCheck:
    def test_hardware_device_passes_exactly(self):
        result = check_device(0)
        assert result.ok, f"failed at {result.stage}: {result.error}"
        assert result.stage == "ok"
        assert result.max_error is not None
        assert result.max_error <= selftest._TOLERANCE

    def test_reports_renderer_and_version(self):
        result = check_device(0)
        assert result.renderer
        assert result.gl_version
        assert result.version_code and result.version_code >= 330

    def test_timings_are_separated(self):
        """Context creation is dominated by one-time driver init, so conflating
        it with shader time would make a fast GPU look slow."""
        result = check_device(0)
        assert result.context_ms > 0
        assert result.shader_ms > 0
        assert result.total_ms == pytest.approx(
            result.context_ms + result.shader_ms
        )

    def test_serialisable(self):
        import json

        payload = check_device(0).to_dict()
        assert json.loads(json.dumps(payload))["ok"] is True


@pytest.mark.cpu
class TestSoftwareDevice:
    def test_software_device_also_passes(self):
        """Software rasterizers are legitimate targets and must verify too --
        and this is a second GLSL compiler checking the same expectations."""
        from shadertoy_local.context import enumerate_devices

        software = [d for d in enumerate_devices() if d.is_software]
        if not software:
            pytest.skip("no software EGL device available")
        result = check_device(software[0].index)
        assert result.ok, f"failed at {result.stage}: {result.error}"
        assert result.software is True
        assert result.max_error <= selftest._TOLERANCE


@pytest.mark.gpu
class TestFailureDetection:
    """Induce each failure mode and confirm it is caught at the right stage."""

    def test_nonexistent_device_fails_at_context(self):
        result = check_device(99)
        assert not result.ok
        assert result.stage == "context"
        assert "does not exist" in result.error

    def test_uncompilable_shader_fails_at_compile(self, restore_fragment):
        restore_fragment(
            selftest._FRAGMENT.replace("gl_FragCoord.xy", "totally_undefined")
        )
        result = check_device(0)
        assert not result.ok
        assert result.stage == "compile"

    def test_wrong_output_fails_at_verify(self, restore_fragment):
        """A 1% error must be caught: rendering *something* is not enough."""
        restore_fragment(
            selftest._FRAGMENT.replace("uv * uScale", "uv * uScale * 1.01")
        )
        result = check_device(0)
        assert not result.ok
        assert result.stage == "verify"
        assert result.max_error > selftest._TOLERANCE

    def test_tiny_error_within_tolerance_still_passes(self, restore_fragment):
        """Legitimate float differences between drivers must not fail."""
        restore_fragment(
            selftest._FRAGMENT.replace("uv * uScale", "uv * uScale + 1e-8")
        )
        result = check_device(0)
        assert result.ok, f"tolerance too tight: {result.error}"

    def test_nan_output_fails_at_verify(self, restore_fragment):
        restore_fragment(
            selftest._FRAGMENT.replace(
                "vec4(uv * uScale, uArray[0].x, 1.0)",
                "vec4(sqrt(-1.0), 0.0, 0.0, 1.0)",
            )
        )
        result = check_device(0)
        assert not result.ok
        assert result.stage == "verify"
        assert "NaN" in result.error

    def test_array_uniform_is_actually_exercised(self, restore_fragment):
        """The blue channel comes from an array uniform, so a driver that
        mishandles array length cannot pass. This is the class of bug that broke
        every Mesa driver."""
        restore_fragment(
            selftest._FRAGMENT.replace("uArray[0].x", "0.0")
        )
        result = check_device(0)
        assert not result.ok, "array uniform is not being verified"
        assert result.stage == "verify"


@pytest.mark.gpu
class TestCheckAllDevices:
    def test_one_result_per_enumerated_device(self):
        from shadertoy_local.context import enumerate_devices

        devices = enumerate_devices()
        results = check_all_devices()
        assert len(results) == len(devices)
        assert {r.device_index for r in results} == {d.index for d in devices}

    def test_every_device_here_works(self):
        for result in check_all_devices():
            assert result.ok, (
                f"device {result.device_index} failed at {result.stage}: "
                f"{result.error}"
            )


class TestRuntimeCheckDataclass:
    def test_summary_when_ok(self):
        check = RuntimeCheck(
            device_index=0,
            ok=True,
            stage="ok",
            gl_version="4.6",
            context_ms=100.0,
            shader_ms=2.0,
        )
        text = check.summary()
        assert "ok" in text and "4.6" in text and "2.00" in text

    def test_summary_when_failed(self):
        check = RuntimeCheck(
            device_index=1, ok=False, stage="compile", error="boom"
        )
        assert "FAILED at compile" in check.summary()
        assert "boom" in check.summary()
