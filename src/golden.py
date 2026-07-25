"""Golden-image regression testing.

References are stored as 8-bit PNGs rather than float dumps, deliberately:
they are small, diffable in a browser, reviewable in a pull request, and immune
to the last-bit floating point differences that vary between drivers.

Comparison reports both the maximum and the mean absolute channel difference.
Max alone is too brittle (one stray pixel fails the run) and mean alone is too
lax (a small bright artefact averages away), so a tolerance can be set on each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import load_png, save_png, to_uint8_image

#: Directory inside a project holding reference images.
GOLDEN_DIR = "golden"

#: Default per-channel tolerances, in 8-bit levels.
DEFAULT_MAX_DIFF = 2
DEFAULT_MEAN_DIFF = 0.5


@dataclass
class Comparison:
    """The result of comparing one rendered frame with its reference."""

    name: str
    golden_path: Path
    status: str  # "pass" | "fail" | "missing" | "size-mismatch"
    max_diff: int | None = None
    mean_diff: float | None = None
    differing_pixels: int | None = None
    total_pixels: int | None = None
    message: str = ""
    diff_path: Path | None = None
    actual_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "golden": str(self.golden_path),
            "max_diff": self.max_diff,
            "mean_diff": self.mean_diff,
            "differing_pixels": self.differing_pixels,
            "total_pixels": self.total_pixels,
            "fraction_differing": (
                self.differing_pixels / self.total_pixels
                if self.differing_pixels is not None and self.total_pixels
                else None
            ),
            "message": self.message,
            "diff_image": str(self.diff_path) if self.diff_path else None,
            "actual_image": str(self.actual_path) if self.actual_path else None,
        }


def golden_path(root: Path, name: str) -> Path:
    """Path of the reference image for a given capture name."""
    return Path(root) / GOLDEN_DIR / f"{name}.png"


def capture_name(pass_name: str, frame: int) -> str:
    """Stable, filesystem-safe name for a captured frame."""
    return f"{pass_name}_f{frame:04d}"


def write_golden(array: np.ndarray, root: Path, name: str) -> Path:
    """Write (or overwrite) a reference image."""
    return save_png(array, golden_path(root, name), opaque=True)


def compare(
    array: np.ndarray,
    root: Path,
    name: str,
    *,
    max_diff: int = DEFAULT_MAX_DIFF,
    mean_diff: float = DEFAULT_MEAN_DIFF,
    write_artifacts: Path | None = None,
) -> Comparison:
    """Compare a rendered frame against its stored reference."""
    path = golden_path(root, name)
    if not path.is_file():
        return Comparison(
            name=name,
            golden_path=path,
            status="missing",
            message=(
                f"no reference image at {path}. Run `shadertoy bless` to create it."
            ),
        )

    actual = to_uint8_image(array, opaque=True)
    expected = load_png(path)

    if actual.shape != expected.shape:
        return Comparison(
            name=name,
            golden_path=path,
            status="size-mismatch",
            message=(
                f"size differs: rendered {actual.shape[1]}x{actual.shape[0]}, "
                f"reference {expected.shape[1]}x{expected.shape[0]}. Re-bless or "
                f"render at the reference size."
            ),
        )

    # int16 avoids uint8 wraparound in the subtraction.
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    # Alpha is forced opaque on both sides, so only RGB is meaningful.
    diff_rgb = diff[..., :3]
    observed_max = int(diff_rgb.max())
    observed_mean = float(diff_rgb.mean())
    per_pixel = diff_rgb.max(axis=2)
    differing = int((per_pixel > 0).sum())
    total = int(per_pixel.size)

    passed = observed_max <= max_diff and observed_mean <= mean_diff
    result = Comparison(
        name=name,
        golden_path=path,
        status="pass" if passed else "fail",
        max_diff=observed_max,
        mean_diff=observed_mean,
        differing_pixels=differing,
        total_pixels=total,
    )
    if not passed:
        result.message = (
            f"max diff {observed_max} (limit {max_diff}), "
            f"mean diff {observed_mean:.3f} (limit {mean_diff}), "
            f"{differing}/{total} pixels differ"
        )
        if write_artifacts is not None:
            result.actual_path, result.diff_path = _write_artifacts(
                write_artifacts, name, actual, diff_rgb
            )
    return result


def _write_artifacts(
    directory: Path, name: str, actual: np.ndarray, diff_rgb: np.ndarray
) -> tuple[Path, Path]:
    """Save the rendered frame and an amplified diff for inspection."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    actual_path = directory / f"{name}.actual.png"
    Image.fromarray(actual, mode="RGBA").save(actual_path)

    # Amplify so single-level differences are actually visible.
    amplified = np.clip(diff_rgb.astype(np.int32) * 16, 0, 255).astype(np.uint8)
    rgba = np.dstack(
        [amplified, np.full(amplified.shape[:2], 255, dtype=np.uint8)]
    )
    diff_path = directory / f"{name}.diff.png"
    Image.fromarray(rgba, mode="RGBA").save(diff_path)
    return actual_path, diff_path
