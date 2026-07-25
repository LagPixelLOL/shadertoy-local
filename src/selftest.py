"""Per-device runtime verification.

``shadertoy info`` can enumerate devices and read their renderer strings without
ever proving they work. That gap is not theoretical: a device may enumerate but
refuse to bind (MIG-partitioned, compute-exclusive, out of memory), bind but fail
to compile, or compile but disagree with other drivers about uniform layout.

So instead of a smoke test, this renders a shader whose exact output is known and
which touches precisely the features the real renderer relies on:

* the vertex path, using the same vertex shader as a real pass;
* ``gl_FragCoord``, i.e. that rasterisation and interpolation happen at all;
* scalar, vector and **array** uniforms -- the array matters, because drivers
  disagree on how many elements of an array uniform they expose, and getting it
  wrong broke every Mesa driver until it was caught;
* an ``RGBA32F`` target holding values above 1.0 without clamping;
* pixel readback.

Every pixel is then compared against values computed on the CPU, so a driver that
renders *something* but renders it wrongly is reported as a failure rather than a
success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .compose import vertex_source
from .context import ContextError, DeviceInfo, create_context, enumerate_devices

#: Edge length of the verification target. Small: this runs per device, and the
#: point is correctness, not throughput.
_SIZE = 8

#: Uniform values chosen so the expected output is easy to reason about and
#: exceeds 1.0, which an 8-bit target would silently clamp.
_SCALE = 2.0
_ARRAY_X = 0.75

#: Tolerance for the pixel comparison. Generous enough for legitimate
#: floating-point differences between drivers, tight enough that a genuinely
#: wrong result cannot pass.
_TOLERANCE = 1e-5

_FRAGMENT = """#version 330 core
uniform vec2  uResolution;
uniform float uScale;
uniform vec3  uArray[4];
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    fragColor = vec4(uv * uScale, uArray[0].x, 1.0);
}
"""


@dataclass
class RuntimeCheck:
    """The outcome of verifying one device."""

    device_index: int | None
    ok: bool
    #: Last stage reached: context, compile, render, verify, or ok.
    stage: str
    renderer: str | None = None
    gl_version: str | None = None
    version_code: int | None = None
    software: bool = False
    #: Time to create the context. Dominated by one-time driver initialisation,
    #: so it says little about the device and is reported separately.
    context_ms: float = 0.0
    #: Time to compile, render, read back and verify -- the informative number.
    shader_ms: float = 0.0
    #: Largest absolute deviation from the expected image.
    max_error: float | None = None
    error: str | None = None

    @property
    def total_ms(self) -> float:
        return self.context_ms + self.shader_ms

    def summary(self) -> str:
        if self.ok:
            detail = self.gl_version or ""
            return (
                f"ok    {detail}  shader {self.shader_ms:.2f} ms"
                f"  (context {self.context_ms:.0f} ms)"
            )
        return f"FAILED at {self.stage}: {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_index": self.device_index,
            "ok": self.ok,
            "stage": self.stage,
            "renderer": self.renderer,
            "gl_version": self.gl_version,
            "version_code": self.version_code,
            "software": self.software,
            "context_ms": round(self.context_ms, 4),
            "shader_ms": round(self.shader_ms, 4),
            "total_ms": round(self.total_ms, 4),
            "max_error": self.max_error,
            "error": self.error,
        }


def expected_image(size: int = _SIZE) -> np.ndarray:
    """The image the check shader must produce, computed on the CPU."""
    # gl_FragCoord lands at pixel centres, and row 0 of a GL readback is the
    # bottom row, which matches this indexing.
    coords = (np.arange(size, dtype=np.float64) + 0.5) / size
    out = np.empty((size, size, 4), dtype=np.float64)
    out[..., 0] = coords[None, :] * _SCALE
    out[..., 1] = coords[:, None] * _SCALE
    out[..., 2] = _ARRAY_X
    out[..., 3] = 1.0
    return out


def check_device(
    device_index: int | None = None,
    *,
    require: int | None = None,
    allow_software: bool = True,
    size: int = _SIZE,
) -> RuntimeCheck:
    """Bind a device, render a known image, and verify it pixel by pixel."""
    from .renderer import _set_uniform

    result = RuntimeCheck(device_index=device_index, ok=False, stage="context")
    started = time.perf_counter()

    try:
        handle = create_context(
            device_index=device_index,
            require=require,
            allow_software=allow_software,
        )
    except ContextError as exc:
        result.context_ms = (time.perf_counter() - started) * 1000.0
        result.error = str(exc).splitlines()[0]
        return result
    result.context_ms = (time.perf_counter() - started) * 1000.0
    shader_started = time.perf_counter()

    try:
        info = handle.ctx.info
        result.renderer = info.get("GL_RENDERER")
        result.gl_version = info.get("GL_VERSION")
        result.version_code = handle.version_code
        result.software = bool(handle.device and handle.device.is_software)
        if handle.device is not None:
            result.device_index = handle.device.index

        ctx = handle.ctx
        objects: list[Any] = []
        try:
            result.stage = "compile"
            program = ctx.program(
                vertex_shader=vertex_source(), fragment_shader=_FRAGMENT
            )
            objects.append(program)

            result.stage = "render"
            quad = ctx.buffer(
                np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
            )
            objects.append(quad)
            vao = ctx.vertex_array(program, [(quad, "2f", "_st_position")])
            objects.append(vao)
            texture = ctx.texture((size, size), 4, dtype="f4")
            objects.append(texture)
            fbo = ctx.framebuffer(color_attachments=[texture])
            objects.append(fbo)

            _set_uniform(program, "uResolution", (float(size), float(size)))
            _set_uniform(program, "uScale", _SCALE)
            # Deliberately supply all four elements even though the shader reads
            # only [0]: drivers that trim the array must still be handled.
            _set_uniform(program, "uArray", [[_ARRAY_X, 0.0, 0.0]] * 4)

            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(mode=ctx.TRIANGLE_STRIP)
            ctx.finish()

            raw = fbo.read(components=4, dtype="f4")
            actual = np.frombuffer(raw, dtype=np.float32).reshape(size, size, 4)

            result.stage = "verify"
            if not np.isfinite(actual).all():
                result.error = "rendered image contains NaN or Inf"
                return result
            deviation = np.abs(actual.astype(np.float64) - expected_image(size))
            result.max_error = float(deviation.max())
            if result.max_error > _TOLERANCE:
                result.error = (
                    f"rendered image differs from expected by "
                    f"{result.max_error:.6g} (tolerance {_TOLERANCE:g})"
                )
                return result

            result.stage = "ok"
            result.ok = True
            return result
        except Exception as exc:  # noqa: BLE001 - any driver failure is a result
            result.error = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
            return result
        finally:
            for item in reversed(objects):
                try:
                    item.release()
                except Exception:  # pragma: no cover
                    pass
            result.shader_ms = (time.perf_counter() - shader_started) * 1000.0
    finally:
        handle.release()


def check_all_devices(
    *, require: int | None = None, size: int = _SIZE
) -> list[RuntimeCheck]:
    """Verify every enumerated device, software included.

    Software devices are checked too: they are legitimate render targets, and a
    broken one should be reported rather than silently offered as a fallback.
    """
    devices: list[DeviceInfo] = enumerate_devices()
    if not devices:
        # No EGL device enumeration (e.g. the GLX backend); check the default.
        return [check_device(None, require=require, size=size)]
    return [
        check_device(dev.index, require=require, allow_software=True, size=size)
        for dev in devices
    ]
