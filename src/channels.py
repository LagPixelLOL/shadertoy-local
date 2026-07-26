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

from .inputs import KEYBOARD_HEIGHT, KEYBOARD_WIDTH, InputState
from .project import (
    BUILTIN_DEFAULT_SIZES,
    BUILTIN_TEXTURES,
    ChannelBinding,
    ProjectError,
    builtin_identity,
    canonical_builtin,
)

#: Fixed seed so procedural textures are byte-identical across runs/machines.
_SEED = 0x5EED1234


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _to_rgba(gray: np.ndarray) -> np.ndarray:
    """Broadcast a single-channel uint8 image to opaque RGBA."""
    size_y, size_x = gray.shape
    out = np.empty((size_y, size_x, 4), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = gray
    out[..., 3] = 255
    return out


def _rgba_noise(size: int) -> np.ndarray:
    return _rng().integers(0, 256, size=(size, size, 4), dtype=np.uint8)


def _gray_noise(size: int) -> np.ndarray:
    return _to_rgba(_rng().integers(0, 256, size=(size, size), dtype=np.uint8))


def _blue_noise(size: int) -> np.ndarray:
    """High-pass filtered white noise.

    Not true void-and-cluster blue noise, but spectrally biased toward high
    frequencies, which is what shaders use it for. Shadertoy's own blue noise is
    a proper generated tile, so results will differ in quality if not in kind.
    """
    base = _rng().random((size, size), dtype=np.float32)
    blur = np.zeros_like(base)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            blur += np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    blur /= 9.0
    high = base - blur
    high -= high.min()
    if high.max() > 0:
        high /= high.max()
    return _to_rgba((high * 255).astype(np.uint8))


def _bayer(size: int) -> np.ndarray:
    """An ordered-dither (Bayer) threshold matrix.

    Unlike every other generator here, this one is *exact*: a Bayer matrix is
    defined by recurrence rather than authored, so a 16x16 tile is bit-identical
    to the one shadertoy.com ships. Built by the standard recurrence
    ``B(2n) = [[4B+0, 4B+2], [4B+3, 4B+1]]``.
    """
    matrix = np.zeros((1, 1), dtype=np.int64)
    while matrix.shape[0] < size:
        matrix = np.block(
            [
                [4 * matrix + 0, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
    matrix = matrix[:size, :size]
    levels = matrix.max() + 1
    # Map thresholds onto the full 0..255 range.
    scaled = (matrix * 255 // max(1, levels - 1)).astype(np.uint8)
    return _to_rgba(scaled)


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
    return _to_rgba(np.broadcast_to(ramp[None, :], (size, size)).copy())


def _solid(size: int, value: int) -> np.ndarray:
    out = np.full((size, size, 4), value, dtype=np.uint8)
    out[..., 3] = 255
    return out


#: name -> (generator, default edge length).
#:
#: The first group mirrors the role and dimensions of textures shadertoy.com
#: provides, so a shader ported from the site samples something of the right kind
#: at the right resolution. The pixels are NOT identical -- those assets cannot
#: be redistributed -- so ``bayer`` aside, expect different values.
#:
#: The second group has no counterpart on the site. They exist for local
#: debugging, and a project using them cannot be reproduced there by binding a
#: stock input.
#: Canonical names only; lookups go through canonical_builtin, and the default
#: sizes live in project.BUILTIN_DEFAULT_SIZES so validation and rendering
#: cannot disagree about what "the same texture" means.
_GENERATORS: dict[str, Any] = {
    # -- approximations of shadertoy.com assets --
    "rgba-noise-small": _rgba_noise,
    "rgba-noise-medium": _rgba_noise,
    "gray-noise-small": _gray_noise,
    "gray-noise-medium": _gray_noise,
    "blue-noise": _blue_noise,
    "bayer": _bayer,
    # -- local-only debug aids (no shadertoy.com equivalent) --
    "checker": _checker,
    "uv": _uv,
    "gradient": _gradient,
    "white": lambda size: _solid(size, 255),
    "black": lambda size: _solid(size, 0),
}

def sampler_filter_modes(ctx: Any, name: str) -> tuple[Any, Any]:
    """Translate a declared filter name into ``(min, mag)`` GL filter modes.

    Exhaustive on purpose. The obvious spelling is nearest/mipmap/else-linear,
    but that silently turns any filter added to ``_FILTERS`` without updating this
    function into linear -- in two separate places, since buffers are sampled
    through sampler objects and textures through the texture itself. Raising keeps
    the failure loud and keeps the mapping in one place.
    """
    if name == "nearest":
        return (ctx.NEAREST, ctx.NEAREST)
    if name == "linear":
        return (ctx.LINEAR, ctx.LINEAR)
    if name == "mipmap":
        return (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
    raise ProjectError(
        f"unsupported filter {name!r}; sampler_filter_modes needs a case for it"
    )


def sampler_repeats(name: str) -> bool:
    """Whether a declared wrap name means repeat. Exhaustive for the same reason."""
    if name == "repeat":
        return True
    if name == "clamp":
        return False
    raise ProjectError(
        f"unsupported wrap {name!r}; sampler_repeats needs a case for it"
    )


def builtin_array(name: str, size: int | None = None) -> np.ndarray:
    """Generate a builtin texture as an ``(h, w, 4)`` uint8 array.

    *size* overrides the default edge length, which is a deliberate escape hatch:
    if one of the assumed shadertoy.com dimensions is wrong, a project can
    correct it without waiting on a code change.
    """
    canonical = canonical_builtin(name)
    try:
        generator = _GENERATORS[canonical]
    except KeyError:
        raise ProjectError(
            f"unknown builtin texture {name!r}. Available: "
            f"{', '.join(sorted(BUILTIN_TEXTURES))}"
        ) from None
    return generator(int(size) if size else BUILTIN_DEFAULT_SIZES[canonical])


def builtin_default_size(name: str) -> int | None:
    """Default edge length for a builtin, or ``None`` if unknown."""
    return BUILTIN_DEFAULT_SIZES.get(canonical_builtin(name))


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

    def keyboard(self, state: InputState, binding: ChannelBinding | None = None) -> Any:
        """The 256x3 keyboard texture, updated in place each frame.

        The state is already resolved for the frame in question, so no frame
        index is needed here. There is exactly one keyboard texture however
        many channels read it, which is also how the site behaves -- and why
        the project loader rejects references that disagree about its filter.
        Wrap is always clamp and the filter is nearest or linear; both are
        enforced at load time, so applying the binding here cannot fight.
        """
        if self._keyboard is None:
            self._keyboard = self.ctx.texture(
                (KEYBOARD_WIDTH, KEYBOARD_HEIGHT), 1, dtype="f1"
            )
            self._keyboard.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
            self._keyboard.repeat_x = False
            self._keyboard.repeat_y = False
            self._owned.append(self._keyboard)
        if binding is not None:
            self._keyboard.filter = sampler_filter_modes(self.ctx, binding.filter)
        self._keyboard.write(state.keyboard_bytes())
        return self._keyboard

    def get(self, binding: ChannelBinding) -> Any:
        """Return the texture for a texture/builtin binding (never a buffer)."""
        if binding.is_buffer:
            raise AssertionError("buffer channels are resolved by the renderer")

        # One cache entry per *object*, not per spelling: "noise" and
        # "rgba-noise-medium" are one asset on the site, as are a builtin
        # with its default size and the same builtin with that size spelled
        # out. Load-time validation guarantees all references to one object
        # agree on the sampler settings, so the settings in the key can never
        # split what the identity joins -- they only keep genuinely different
        # objects (two files, two sizes) apart.
        if binding.is_builtin:
            identity: tuple = builtin_identity(binding.source, binding.size)
        else:
            identity = (str(binding.path),)
        key = (
            *identity,
            binding.filter,
            binding.wrap,
            binding.vflip,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if binding.is_builtin:
            array = builtin_array(binding.source, binding.size)
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
        if binding.filter == "mipmap":
            texture.build_mipmaps()
        texture.filter = sampler_filter_modes(ctx, binding.filter)
        repeat = sampler_repeats(binding.wrap)
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
