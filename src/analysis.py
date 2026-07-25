"""Frame analysis: statistics, pixel probes, and PNG encoding.

This is what makes the tool usable without looking at pictures. An agent cannot
see a PNG, but it can assert that a frame is not uniformly black, contains no
NaNs, and that the pixel at (320, 180) is approximately red.

All arrays are float32 ``(h, w, 4)`` in GL order, so row 0 is the *bottom* row.
Probe coordinates therefore use Shadertoy's ``fragCoord`` convention directly
(origin bottom-left), while PNG output flips to top-down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_CHANNEL_NAMES = ("r", "g", "b", "a")


class AnalysisError(ValueError):
    """Raised for malformed probe specifications."""


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def frame_stats(array: np.ndarray) -> dict[str, Any]:
    """Summarise a frame in a form that is cheap to assert against."""
    height, width = array.shape[:2]
    finite_mask = np.isfinite(array)
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    finite = array[finite_mask]

    per_channel = {}
    for index, name in enumerate(_CHANNEL_NAMES):
        channel = array[..., index]
        good = channel[np.isfinite(channel)]
        per_channel[name] = {
            "min": float(good.min()) if good.size else None,
            "max": float(good.max()) if good.size else None,
            "mean": float(good.mean()) if good.size else None,
        }

    rgb = array[..., :3]
    rgb_finite = np.where(np.isfinite(rgb), rgb, 0.0)
    # Rec. 709 luma, the usual perceptual weighting.
    luma = (
        0.2126 * rgb_finite[..., 0]
        + 0.7152 * rgb_finite[..., 1]
        + 0.0722 * rgb_finite[..., 2]
    )

    quantised = np.clip(rgb_finite, 0.0, 1.0)
    quantised = np.rint(quantised * 255).astype(np.uint8)
    unique_colors = int(len(np.unique(quantised.reshape(-1, 3), axis=0)))

    first = rgb_finite.reshape(-1, 3)[0]
    is_uniform = bool(np.allclose(rgb_finite, first, atol=1e-6))

    return {
        "width": int(width),
        "height": int(height),
        "pixels": int(width * height),
        "channels": per_channel,
        "luma": {
            "min": float(luma.min()),
            "max": float(luma.max()),
            "mean": float(luma.mean()),
        },
        "unique_colors": unique_colors,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "has_nan": nan_count > 0,
        "has_inf": inf_count > 0,
        "finite": nan_count == 0 and inf_count == 0,
        "is_uniform": is_uniform,
        "is_black": bool(finite.size and float(np.abs(rgb_finite).max()) == 0.0),
        # Fractions outside the displayable range: useful for spotting shaders
        # that look fine only because the display clamps them.
        "fraction_negative": float((rgb_finite < 0.0).mean()),
        "fraction_above_one": float((rgb_finite > 1.0).mean()),
        "fraction_clipped": float(
            ((rgb_finite <= 0.0) | (rgb_finite >= 1.0)).mean()
        ),
    }


def histogram(array: np.ndarray, bins: int = 16) -> dict[str, list[int]]:
    """Per-channel histogram over the [0, 1] range."""
    out: dict[str, list[int]] = {}
    for index, name in enumerate(_CHANNEL_NAMES):
        channel = array[..., index]
        channel = channel[np.isfinite(channel)]
        counts, _ = np.histogram(np.clip(channel, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
        out[name] = [int(c) for c in counts]
    return out


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


@dataclass
class Probe:
    """A pixel query, optionally with an expected value."""

    x: float
    y: float
    normalized: bool = False
    expect: tuple[float, ...] | None = None
    tolerance: float = 1 / 255

    def resolve(self, width: int, height: int) -> tuple[int, int]:
        x, y = self.x, self.y
        if self.normalized:
            x *= width
            y *= height
        # Sample the pixel centre containing the coordinate.
        ix = int(np.clip(np.floor(x), 0, width - 1))
        iy = int(np.clip(np.floor(y), 0, height - 1))
        return ix, iy


def parse_probe(text: str) -> Probe:
    """Parse ``X,Y`` or ``X,Y=R,G,B[,A]`` probe syntax.

    Coordinates ending in ``%`` (or values in [0,1] with an explicit ``n``
    prefix) are treated as normalised.
    """
    raw = str(text).strip()
    if not raw:
        raise AnalysisError("empty probe specification")

    expect: tuple[float, ...] | None = None
    if "=" in raw:
        raw, _, expect_text = raw.partition("=")
        parts = [p.strip() for p in expect_text.split(",") if p.strip()]
        if not 1 <= len(parts) <= 4:
            raise AnalysisError(
                f"probe expectation needs 1 to 4 components (got {expect_text!r})"
            )
        try:
            expect = tuple(float(p) for p in parts)
        except ValueError:
            raise AnalysisError(
                f"probe expectation must be numbers (got {expect_text!r})"
            ) from None

    coords = raw.strip()
    normalized = False
    if coords.startswith("n:"):
        normalized = True
        coords = coords[2:]
    parts = [p.strip() for p in coords.split(",")]
    if len(parts) != 2:
        raise AnalysisError(f"probe needs 'X,Y' coordinates (got {raw!r})")

    values: list[float] = []
    for part in parts:
        if part.endswith("%"):
            normalized = True
            part = part[:-1]
        try:
            values.append(float(part))
        except ValueError:
            raise AnalysisError(f"probe coordinate is not a number: {part!r}") from None
    if normalized and any(p.endswith("%") for p in parts):
        values = [v / 100.0 for v in values]

    return Probe(x=values[0], y=values[1], normalized=normalized, expect=expect)


def run_probe(array: np.ndarray, probe: Probe) -> dict[str, Any]:
    """Evaluate a probe against a frame."""
    height, width = array.shape[:2]
    x, y = probe.resolve(width, height)
    pixel = array[y, x]
    # Non-finite values cannot be rounded to an integer, so sanitise for the
    # 8-bit view only; the float view and the `finite` flag keep the truth.
    displayable = np.nan_to_num(
        np.asarray(pixel, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
    )
    result: dict[str, Any] = {
        "x": x,
        "y": y,
        "rgba": [float(v) for v in pixel],
        "rgba8": [int(np.clip(round(float(v) * 255), 0, 255)) for v in displayable],
        "finite": bool(np.isfinite(pixel).all()),
    }
    if probe.expect is not None:
        expected = np.asarray(probe.expect, dtype=np.float64)
        actual = np.asarray(pixel[: len(expected)], dtype=np.float64)
        diff = np.abs(actual - expected)
        result["expected"] = [float(v) for v in expected]
        result["max_diff"] = float(diff.max())
        result["tolerance"] = probe.tolerance
        result["passed"] = bool(np.isfinite(diff).all() and diff.max() <= probe.tolerance)
    return result


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def to_uint8_image(array: np.ndarray, *, opaque: bool = True) -> np.ndarray:
    """Convert a float GL-order frame to a top-down 8-bit RGBA image.

    Non-finite values become 0 so a broken shader yields a writable file rather
    than an exception; ``frame_stats`` is the right place to detect them.
    """
    data = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    data = np.clip(data, 0.0, 1.0)
    out = np.rint(data * 255.0).astype(np.uint8)
    if opaque:
        out = out.copy()
        out[..., 3] = 255
    # GL row 0 is the bottom row; image formats expect the top row first.
    return np.flipud(out)


def save_png(array: np.ndarray, path: Path, *, opaque: bool = True) -> Path:
    """Write a frame to a PNG."""
    from PIL import Image

    image = to_uint8_image(array, opaque=opaque)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGBA").save(path)
    return path


def load_png(path: Path) -> np.ndarray:
    """Load a PNG as a top-down uint8 RGBA array."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)
