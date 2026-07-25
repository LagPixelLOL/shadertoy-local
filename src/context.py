"""GL context creation.

Headless rendering here goes through EGL's device platform
(``EGL_EXT_platform_device``), which needs neither an X server nor a
``/dev/dri`` node. Proprietary NVIDIA drivers talk to ``/dev/nvidia*``
directly, so this works in containers and sandboxes where DRM is absent.

The main jobs of this module:

* enumerate EGL devices and *prefer real hardware over software rasterizers*
  (llvmpipe is frequently enumerated first, which would silently make every
  render 100x slower);
* turn the notoriously opaque EGL setup failures into actionable messages.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

# EGL enums we need (avoids a hard dependency on any EGL header/binding).
_EGL_EXTENSIONS = 0x3055
_EGL_DRM_DEVICE_FILE_EXT = 0x3233
_EGL_RENDERER_EXT = 0x335F

#: GL versions tried in order when the caller does not pin one.
_VERSION_LADDER = (460, 450, 430, 410, 400, 330)


class ContextError(RuntimeError):
    """Raised when a usable GL context cannot be created."""


@dataclass
class DeviceInfo:
    """A single EGL device as reported by ``eglQueryDevicesEXT``."""

    index: int
    extensions: tuple[str, ...] = ()
    drm_device: str | None = None
    renderer: str | None = None

    @property
    def is_software(self) -> bool:
        return "EGL_MESA_device_software" in self.extensions

    @property
    def is_nvidia(self) -> bool:
        return "EGL_NV_device_cuda" in self.extensions

    @property
    def label(self) -> str:
        if self.renderer:
            return self.renderer
        if self.is_software:
            return "software rasterizer"
        if self.is_nvidia:
            return "NVIDIA device"
        return "unknown device"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "renderer": self.renderer,
            "software": self.is_software,
            "drm_device": self.drm_device,
            "extensions": list(self.extensions),
        }


@dataclass
class ContextHandle:
    """A live moderngl context plus the device it was created on."""

    ctx: Any
    device: DeviceInfo | None
    backend: str
    version_code: int

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # pragma: no cover - driver teardown is best effort
            pass

    def to_dict(self) -> dict[str, Any]:
        info = self.ctx.info
        return {
            "backend": self.backend,
            "device_index": self.device.index if self.device else None,
            "version_code": self.version_code,
            "gl_version": info.get("GL_VERSION"),
            "gl_renderer": info.get("GL_RENDERER"),
            "gl_vendor": info.get("GL_VENDOR"),
            # moderngl does not surface GL_SHADING_LANGUAGE_VERSION; for GL >= 3.3
            # the GLSL version number matches the GL version number.
            "glsl_version": self.version_code,
            "max_texture_size": info.get("GL_MAX_TEXTURE_SIZE"),
            "software": bool(self.device and self.device.is_software),
        }


# --------------------------------------------------------------------------
# EGL device enumeration
# --------------------------------------------------------------------------


def _load_egl() -> ctypes.CDLL | None:
    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _egl_proc(egl: ctypes.CDLL, name: str, restype, *argtypes):
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    addr = egl.eglGetProcAddress(name.encode())
    if not addr:
        return None
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)


def enumerate_devices() -> list[DeviceInfo]:
    """Return every EGL device, or ``[]`` if EGL device enumeration is absent."""
    egl = _load_egl()
    if egl is None:
        return []

    device_t = ctypes.c_void_p
    query_devices = _egl_proc(
        egl,
        "eglQueryDevicesEXT",
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(device_t),
        ctypes.POINTER(ctypes.c_int),
    )
    if query_devices is None:
        return []

    count = ctypes.c_int(0)
    if not query_devices(0, None, ctypes.byref(count)) or count.value <= 0:
        return []

    handles = (device_t * count.value)()
    if not query_devices(count.value, handles, ctypes.byref(count)):
        return []

    query_string = _egl_proc(
        egl, "eglQueryDeviceStringEXT", ctypes.c_char_p, device_t, ctypes.c_int
    )

    devices: list[DeviceInfo] = []
    for index, handle in enumerate(handles[: count.value]):
        exts: tuple[str, ...] = ()
        drm = None
        renderer = None
        if query_string is not None:
            raw = query_string(handle, _EGL_EXTENSIONS)
            if raw:
                exts = tuple(sorted(raw.decode(errors="replace").split()))
            # Only valid when EGL_EXT_device_drm is advertised; returns NULL
            # when no DRM node is associated (e.g. nvidia-drm not loaded).
            if "EGL_EXT_device_drm" in exts:
                raw = query_string(handle, _EGL_DRM_DEVICE_FILE_EXT)
                drm = raw.decode(errors="replace") if raw else None
            if "EGL_EXT_device_query_name" in exts:
                raw = query_string(handle, _EGL_RENDERER_EXT)
                renderer = raw.decode(errors="replace") if raw else None
        devices.append(
            DeviceInfo(index=index, extensions=exts, drm_device=drm, renderer=renderer)
        )
    return devices


def select_device(
    devices: list[DeviceInfo], requested: int | None = None, allow_software: bool = False
) -> DeviceInfo | None:
    """Pick a device, preferring hardware unless told otherwise."""
    if not devices:
        return None
    if requested is not None:
        matches = [d for d in devices if d.index == requested]
        if not matches:
            available = ", ".join(str(d.index) for d in devices)
            raise ContextError(
                f"EGL device index {requested} does not exist (available: {available})"
            )
        return matches[0]

    hardware = [d for d in devices if not d.is_software]
    if hardware:
        # A CUDA-capable device is the most likely intended GPU.
        hardware.sort(key=lambda d: (not d.is_nvidia, d.index))
        return hardware[0]
    if allow_software:
        return devices[0]
    return None


# --------------------------------------------------------------------------
# Context creation
# --------------------------------------------------------------------------


def _egl_setup_hint() -> str:
    """Diagnose *why* EGL is unusable and say how to fix it."""
    problems: list[str] = []
    if _load_egl() is None:
        problems.append(
            "libEGL.so.1 is missing (the GLVND dispatch library).\n"
            "  Debian/Ubuntu: apt-get install -y --no-install-recommends libegl1"
        )
    vendor_dir = "/usr/share/glvnd/egl_vendor.d"
    if os.path.isdir(vendor_dir):
        entries = [e for e in os.listdir(vendor_dir) if e.endswith(".json")]
        has_nvidia_lib = any(
            os.path.exists(f"/usr/lib/{arch}/libEGL_nvidia.so.0")
            for arch in ("x86_64-linux-gnu", "aarch64-linux-gnu")
        )
        if has_nvidia_lib and not any("nvidia" in e for e in entries):
            problems.append(
                "libEGL_nvidia.so.0 is installed but has no GLVND vendor config, so\n"
                "  EGL cannot see the NVIDIA GPU. Create "
                f"{vendor_dir}/10_nvidia.json containing:\n"
                '    {"file_format_version":"1.0.0",'
                '"ICD":{"library_path":"libEGL_nvidia.so.0"}}'
            )
    else:
        problems.append(f"{vendor_dir} does not exist; no EGL vendor drivers registered.")
    if not problems:
        return ""
    return "\n\nLikely cause:\n  " + "\n  ".join(problems)


def create_context(
    device_index: int | None = None,
    require: int | None = None,
    backend: str | None = None,
    allow_software: bool = False,
) -> ContextHandle:
    """Create a standalone (offscreen) GL context.

    Args:
        device_index: EGL device to use; ``None`` auto-selects hardware.
        require: Exact GL version code to demand (e.g. ``430``). When ``None``,
            the highest version that succeeds is used.
        backend: moderngl backend; defaults to ``egl`` (override with
            ``SHADERTOY_BACKEND``). GLX needs a running X server.
        allow_software: Permit falling back to a software rasterizer.
    """
    import moderngl

    backend = backend or os.environ.get("SHADERTOY_BACKEND", "egl")
    if device_index is None:
        env_device = os.environ.get("SHADERTOY_DEVICE")
        if env_device:
            try:
                device_index = int(env_device)
            except ValueError:
                raise ContextError(
                    f"SHADERTOY_DEVICE must be an integer, got {env_device!r}"
                ) from None
    if os.environ.get("SHADERTOY_ALLOW_SOFTWARE") == "1":
        allow_software = True

    device: DeviceInfo | None = None
    kwargs: dict[str, Any] = {"standalone": True}
    if backend:
        kwargs["backend"] = backend

    if backend == "egl":
        devices = enumerate_devices()
        if devices:
            device = select_device(devices, device_index, allow_software)
            if device is None:
                listing = "\n  ".join(f"[{d.index}] {d.label}" for d in devices)
                raise ContextError(
                    "Only software EGL devices were found; refusing to render on a\n"
                    "CPU rasterizer. Pass --allow-software to override.\n"
                    f"  {listing}"
                )
            kwargs["device_index"] = device.index

    versions = (require,) if require else _VERSION_LADDER
    errors: list[str] = []
    for version in versions:
        try:
            ctx = moderngl.create_context(require=version, **kwargs)
        except Exception as exc:  # noqa: BLE001 - driver errors are arbitrary
            errors.append(f"require={version}: {exc}")
            continue
        return ContextHandle(
            ctx=ctx, device=device, backend=backend, version_code=ctx.version_code
        )

    detail = "\n  ".join(errors)
    hint = _egl_setup_hint() if backend == "egl" else ""
    if backend != "egl" and "XOpenDisplay" in detail:
        hint = (
            "\n\nLikely cause:\n  The GLX backend needs an X server but DISPLAY is unset."
            "\n  Use the default EGL backend for headless rendering."
        )
    raise ContextError(f"Could not create a GL context.\n  {detail}{hint}")


def probe_devices(require: int | None = None) -> list[DeviceInfo]:
    """Enumerate devices and fill in ``renderer`` by briefly binding each one.

    Used by ``shadertoy info``; creating a context is the only fully reliable
    way to learn a device's real GL renderer string.
    """
    devices = enumerate_devices()
    for dev in devices:
        if dev.renderer:
            continue
        try:
            handle = create_context(
                device_index=dev.index, require=require, allow_software=True
            )
        except Exception:  # noqa: BLE001
            dev.renderer = None
            continue
        try:
            dev.renderer = handle.ctx.info.get("GL_RENDERER")
        finally:
            handle.release()
    return devices
