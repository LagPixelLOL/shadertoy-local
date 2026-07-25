"""Frame statistics, probes, PNG encoding, and golden comparison."""

from __future__ import annotations

import numpy as np
import pytest

from shadertoy_local.analysis import (
    AnalysisError,
    frame_stats,
    histogram,
    load_png,
    parse_probe,
    run_probe,
    save_png,
    to_uint8_image,
)
from shadertoy_local.golden import capture_name, compare, write_golden


def _frame(height=4, width=4, value=0.5):
    return np.full((height, width, 4), value, dtype=np.float32)


class TestStats:
    def test_basic_shape_and_ranges(self):
        stats = frame_stats(_frame(8, 16, 0.25))
        assert (stats["width"], stats["height"]) == (16, 8)
        assert stats["pixels"] == 128
        assert stats["channels"]["r"]["mean"] == pytest.approx(0.25)

    def test_uniform_and_black_detection(self):
        assert frame_stats(_frame(value=0.0))["is_black"]
        assert frame_stats(_frame(value=0.5))["is_uniform"]
        assert not frame_stats(_frame(value=0.5))["is_black"]

    def test_gradient_is_not_uniform(self):
        array = np.zeros((4, 4, 4), dtype=np.float32)
        array[..., 0] = np.linspace(0, 1, 4)[None, :]
        assert not frame_stats(array)["is_uniform"]

    def test_nan_is_detected_and_counted(self):
        """The whole reason targets are float rather than 8-bit."""
        array = _frame()
        array[0, 0, 0] = np.nan
        stats = frame_stats(array)
        assert stats["has_nan"] and stats["nan_count"] == 1
        assert not stats["finite"]

    def test_inf_is_detected(self):
        array = _frame()
        array[1, 1, 2] = np.inf
        stats = frame_stats(array)
        assert stats["has_inf"] and stats["inf_count"] == 1

    def test_stats_survive_all_nan(self):
        array = np.full((2, 2, 4), np.nan, dtype=np.float32)
        stats = frame_stats(array)
        assert stats["nan_count"] == 16
        assert stats["channels"]["r"]["min"] is None

    def test_out_of_range_fractions(self):
        array = np.zeros((2, 2, 4), dtype=np.float32)
        array[0, :, :3] = 2.0     # above one
        array[1, :, :3] = -1.0    # negative
        stats = frame_stats(array)
        assert stats["fraction_above_one"] == pytest.approx(0.5)
        assert stats["fraction_negative"] == pytest.approx(0.5)

    def test_unique_colors(self):
        array = np.zeros((2, 2, 4), dtype=np.float32)
        array[0, 0, 0] = 1.0
        array[0, 1, 1] = 1.0
        assert frame_stats(array)["unique_colors"] == 3  # black, red, green

    def test_histogram_bins_sum_to_pixel_count(self):
        hist = histogram(_frame(4, 4), bins=8)
        assert len(hist["r"]) == 8
        assert sum(hist["r"]) == 16


class TestProbeParsing:
    def test_plain_coordinates(self):
        probe = parse_probe("10,20")
        assert (probe.x, probe.y) == (10.0, 20.0)
        assert probe.expect is None

    def test_with_expectation(self):
        probe = parse_probe("10,20=1,0,0")
        assert probe.expect == (1.0, 0.0, 0.0)

    def test_rgba_expectation(self):
        assert parse_probe("0,0=0.1,0.2,0.3,0.4").expect == (0.1, 0.2, 0.3, 0.4)

    def test_normalized_prefix(self):
        probe = parse_probe("n:0.5,0.5")
        assert probe.normalized
        assert probe.resolve(100, 200) == (50, 100)

    def test_percent_suffix(self):
        probe = parse_probe("50%,25%")
        assert probe.normalized
        assert probe.resolve(100, 200) == (50, 50)

    def test_clamps_inside_bounds(self):
        assert parse_probe("999,999").resolve(10, 10) == (9, 9)

    @pytest.mark.parametrize("bad", ["", "10", "a,b", "10,20=", "1,2=1,2,3,4,5"])
    def test_invalid_specs(self, bad):
        with pytest.raises(AnalysisError):
            parse_probe(bad)


class TestProbeEvaluation:
    def test_reads_expected_pixel(self):
        array = np.zeros((4, 4, 4), dtype=np.float32)
        array[1, 2] = [1.0, 0.5, 0.25, 1.0]
        result = run_probe(array, parse_probe("2,1"))
        assert result["rgba"] == pytest.approx([1.0, 0.5, 0.25, 1.0])
        assert result["rgba8"] == [255, 128, 64, 255]

    def test_passing_expectation(self):
        array = np.zeros((2, 2, 4), dtype=np.float32)
        array[0, 0] = [1.0, 0.0, 0.0, 1.0]
        assert run_probe(array, parse_probe("0,0=1,0,0"))["passed"]

    def test_failing_expectation(self):
        array = np.zeros((2, 2, 4), dtype=np.float32)
        result = run_probe(array, parse_probe("0,0=1,0,0"))
        assert not result["passed"]
        assert result["max_diff"] == pytest.approx(1.0)

    def test_nan_never_passes(self):
        array = np.full((2, 2, 4), np.nan, dtype=np.float32)
        result = run_probe(array, parse_probe("0,0=0,0,0"))
        assert not result["passed"]
        assert not result["finite"]


class TestEncoding:
    def test_flips_to_top_down(self):
        """GL row 0 is the bottom row; PNG row 0 is the top."""
        array = np.zeros((2, 1, 4), dtype=np.float32)
        array[0, 0] = [1.0, 0.0, 0.0, 1.0]   # bottom row red
        image = to_uint8_image(array)
        assert tuple(image[1, 0][:3]) == (255, 0, 0)  # ends up at the bottom
        assert tuple(image[0, 0][:3]) == (0, 0, 0)

    def test_clamps_and_quantises(self):
        array = np.array([[[2.0, -1.0, 0.5, 1.0]]], dtype=np.float32)
        assert list(to_uint8_image(array)[0, 0]) == [255, 0, 128, 255]

    def test_non_finite_becomes_writable(self):
        array = np.array([[[np.nan, np.inf, -np.inf, 1.0]]], dtype=np.float32)
        assert list(to_uint8_image(array)[0, 0]) == [0, 255, 0, 255]

    def test_opaque_flag(self):
        array = np.zeros((1, 1, 4), dtype=np.float32)
        assert to_uint8_image(array, opaque=True)[0, 0, 3] == 255
        assert to_uint8_image(array, opaque=False)[0, 0, 3] == 0

    def test_png_round_trip(self, tmp_path):
        array = np.zeros((4, 8, 4), dtype=np.float32)
        array[..., 0] = np.linspace(0, 1, 8)[None, :]
        path = save_png(array, tmp_path / "sub" / "x.png")
        assert path.is_file()
        assert load_png(path).shape == (4, 8, 4)


class TestGolden:
    def test_capture_name_is_sortable(self):
        assert capture_name("image", 7) == "image_f0007"
        assert capture_name("buffer_a", 0) < capture_name("buffer_a", 1)

    def test_missing_reference(self, tmp_path):
        result = compare(_frame(), tmp_path, "image_f0000")
        assert result.status == "missing"
        assert not result.passed
        assert "bless" in result.message

    def test_identical_passes(self, tmp_path):
        array = _frame(8, 8, 0.4)
        write_golden(array, tmp_path, "image_f0000")
        result = compare(array, tmp_path, "image_f0000")
        assert result.passed
        assert result.max_diff == 0

    def test_small_difference_within_tolerance(self, tmp_path):
        write_golden(_frame(8, 8, 0.5), tmp_path, "k")
        # 0.5 -> 128, 0.502 -> 128 as well; nudge by one level instead.
        result = compare(_frame(8, 8, 0.5 + 1 / 255), tmp_path, "k", max_diff=2)
        assert result.passed

    def test_large_difference_fails(self, tmp_path):
        write_golden(_frame(8, 8, 0.2), tmp_path, "k")
        result = compare(_frame(8, 8, 0.8), tmp_path, "k")
        assert not result.passed
        assert result.max_diff > 2
        assert result.differing_pixels == 64

    def test_size_mismatch_is_distinguished(self, tmp_path):
        write_golden(_frame(4, 4), tmp_path, "k")
        result = compare(_frame(8, 8), tmp_path, "k")
        assert result.status == "size-mismatch"

    def test_mean_tolerance_catches_localized_artifact(self, tmp_path):
        """A few very wrong pixels must fail even though the mean stays low."""
        base = _frame(32, 32, 0.5)
        write_golden(base, tmp_path, "k")
        broken = base.copy()
        broken[0, 0] = [1.0, 1.0, 1.0, 1.0]
        result = compare(broken, tmp_path, "k", max_diff=2, mean_diff=10.0)
        assert not result.passed, "max-diff should catch a single bright pixel"

    def test_artifacts_written_on_failure(self, tmp_path):
        write_golden(_frame(4, 4, 0.1), tmp_path, "k")
        artifacts = tmp_path / "artifacts"
        result = compare(
            _frame(4, 4, 0.9), tmp_path, "k", write_artifacts=artifacts
        )
        assert not result.passed
        assert result.diff_path and result.diff_path.is_file()
        assert result.actual_path and result.actual_path.is_file()

    def test_result_is_serialisable(self, tmp_path):
        import json

        write_golden(_frame(), tmp_path, "k")
        payload = compare(_frame(), tmp_path, "k").to_dict()
        assert json.loads(json.dumps(payload))["passed"] is True
