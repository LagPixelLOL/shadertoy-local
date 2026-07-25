"""Channel inputs: image files, procedural builtins, and the keyboard.

Shadertoy's own texture assets cannot be redistributed, so instead of shipping
lookalikes this module generates a set of deterministic procedural textures.
Determinism matters: golden-image tests would be worthless if ``noise`` differed
between runs, so every generator is seeded from a fixed constant.

Buffer channels are not handled here -- those are resolved by the renderer,
which owns the ping-pong targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .inputs import KEYBOARD_HEIGHT, KEYBOARD_WIDTH, KeyboardState
from .project import BUILTIN_TEXTURES, ChannelBinding, ProjectError

#: Fixed seed so procedural textures are byte-identical across runs/machines.
_SEED = 0x5EED1234
#: Default edge length for generated textures.
_BUILTIN_SIZE = 256


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _noise_rgba(size: int) -> np.ndarray:
    return _rng().integers(0, 256, size=(size, size, 4), dtype=np.uint8)


def _noise_gray(size: int) -> np.ndarray:
    gray = _rng().integers(0, 256, size=(size, size), dtype=np.uint8)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = gray
    out[..., 3] = 255
    return out


def _blue_noise(size: int) -> np.ndarray:
    """A cheap high-pass-filtered noise; not true blue noise but spectrally
    biased toward high frequencies, which is what shaders use it for."""
    base = _rng().random((size, size), dtype=np.float32)
    # Subtract a 3x3 box blur to suppress low frequencies.
    blur = np.zeros_like(base)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            blur += np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    blur /= 9.0
    high = base - blur
    high -= high.min()
    if high.max() > 0:
        high /= high.max()
    gray = (high * 255).astype(np.uint8)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = gray
    out[..., 3] = 255
    return out


def _checker(size: int, cells: int = 8) -> np.ndarray:
    step = max(1, size // cells)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = ((xx // step) + (yy // step)) % 2 == 0
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.where(mask[..., None], 235, 30)
    out[..., 3] = 255
    return out


def _uv(size: int) -> np.ndarray:
    """Red = u, green = v. Invaluable for checking orientation and wrapping."""
    ramp = (np.arange(size, dtype=np.float32) / max(1, size - 1) * 255).astype(np.uint8)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 0] = ramp[None, :]
    out[..., 1] = ramp[:, None]
    out[..., 3] = 255
    return out


def _gradient(size: int) -> np.ndarray:
    ramp = (np.arange(size, dtype=np.float32) / max(1, size - 1) * 255).astype(np.uint8)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = ramp[None, :]
    out[..., 3] = 255
    return out


def _solid(size: int, value: int) -> np.ndarray:
    out = np.full((size, size, 4), value, dtype=np.uint8)
    out[..., 3] = 255
    return out


_GENERATORS = {
    "noise": _noise_rgba,
    "rgba-noise": _noise_rgba,
    "gray-noise": _noise_gray,
    "blue-noise": _blue_noise,
    "checker": _checker,
    "uv": _uv,
    "gradient": _gradient,
    "white": lambda size: _solid(size, 255),
    "black": lambda size: _solid(size, 0),
}


def builtin_array(name: str, size: int = _BUILTIN_SIZE) -> np.ndarray:
    """Generate a builtin texture as an ``(h, w, 4)`` uint8 array."""
    try:
        generator = _GENERATORS[name]
    except KeyError:
        raise ProjectError(
            f"unknown builtin texture {name!r}. Available: "
            f"{', '.join(sorted(BUILTIN_TEXTURES))}"
        ) from None
    return generator(size)


def load_image_array(path: Path) -> np.ndarray:
    """Load an image file as an ``(h, w, 4)`` uint8 array."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ProjectError("Pillow is required to load texture files") from exc
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGBA"), dtype=np.uint8)
    except OSError as exc:
        raise ProjectError(f"could not read texture {path}: {exc}") from exc


class ChannelTextures:
    """Creates and caches the GL textures backing non-buffer channels."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._cache: dict[tuple, Any] = {}
        self._keyboard: Any | None = None
        self._owned: list[Any] = []

    def keyboard(self, state: KeyboardState, frame: int) -> Any:
        """The 256x3 keyboard texture, updated in place each frame."""
        if self._keyboard is None:
            self._keyboard = self.ctx.texture(
                (KEYBOARD_WIDTH, KEYBOARD_HEIGHT), 1, dtype="f1"
            )
            self._keyboard.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
            self._keyboard.repeat_x = False
            self._keyboard.repeat_y = False
            self._owned.append(self._keyboard)
        self._keyboard.write(state.texture_bytes(frame))
        return self._keyboard

    def get(self, binding: ChannelBinding) -> Any:
        """Return the texture for a texture/builtin binding (never a buffer)."""
        if binding.is_buffer:
            raise AssertionError("buffer channels are resolved by the renderer")

        key = (
            binding.source,
            str(binding.path),
            binding.filter,
            binding.wrap,
            binding.vflip,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if binding.is_builtin:
            array = builtin_array(binding.source)
        else:
            assert binding.path is not None
            array = load_image_array(binding.path)

        if binding.vflip:
            # GL texture row 0 is the bottom row; image row 0 is the top.
            array = np.flipud(array)
        array = np.ascontiguousarray(array)

        height, width = array.shape[:2]
        texture = self.ctx.texture((width, height), 4, array.tobytes())
        self._apply_sampling(texture, binding)
        self._cache[key] = texture
        self._owned.append(texture)
        return texture

    def _apply_sampling(self, texture: Any, binding: ChannelBinding) -> None:
        ctx = self.ctx
        if binding.filter == "nearest":
            texture.filter = (ctx.NEAREST, ctx.NEAREST)
        elif binding.filter == "mipmap":
            texture.build_mipmaps()
            texture.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
        else:
            texture.filter = (ctx.LINEAR, ctx.LINEAR)
        repeat = binding.wrap == "repeat"
        texture.repeat_x = repeat
        texture.repeat_y = repeat

    def release(self) -> None:
        for texture in self._owned:
            try:
                texture.release()
            except Exception:  # pragma: no cover
                pass
        self._owned.clear()
        self._cache.clear()
        self._keyboard = None
