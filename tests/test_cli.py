"""End-to-end CLI tests: exit codes and JSON contract.

Exit codes and the JSON payload are the actual interface for scripted use, so
they are tested as deliberately as the rendering itself.
"""

from __future__ import annotations

import json

import pytest

from shadertoy_local.cli import (
    EXIT_ENVIRONMENT,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    _timing_lines,
    main,
)
from shadertoy_local.renderer import RunTiming

BROKEN = "void mainImage(out vec4 c, in vec2 f){ c = does_not_exist; }\n"
BLACK = "void mainImage(out vec4 c, in vec2 f){ c = vec4(0.0,0.0,0.0,1.0); }\n"
NAN_SHADER = "void mainImage(out vec4 c, in vec2 f){ c = vec4(sqrt(-1.0),0,0,1); }\n"


def run(capsys, *argv):
    """Invoke the CLI, returning (exit_code, parsed_json_or_None, stderr)."""
    code = main(list(argv))
    captured = capsys.readouterr()
    payload = None
    if captured.out.strip():
        try:
            payload = json.loads(captured.out)
        except json.JSONDecodeError:
            payload = None
    return code, payload, captured.err


class TestUsage:
    def test_no_command_prints_help(self, capsys):
        code, _, err = run(capsys)
        assert code == EXIT_USAGE
        assert "usage" in err.lower()

    def test_missing_project(self, capsys, tmp_path):
        code, payload, _ = run(capsys, "check", "-C", str(tmp_path), "--json")
        assert code == EXIT_USAGE
        assert payload["ok"] is False
        assert "No Shadertoy project" in payload["error"]

    def test_bad_resolution(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "not-a-size", "--json"
        )
        assert code == EXIT_USAGE
        assert payload["ok"] is False

    def test_bad_key_name(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "--json",
            "--input", '[{"frame": 0, "op": "key_down", "keys": ["nonsense"]}]',
        )
        assert code == EXIT_USAGE
        assert "unknown key" in payload["error"]

    def test_off_canvas_mouse_position(self, capsys, make_project, simple_image):
        """The CLI is where the resolution is known, so this is where it fires."""
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "320x240", "--json",
            "--input", '[{"frame": 0, "op": "mouse_down", "pos": [69420, -69]}]',
        )
        assert code == EXIT_USAGE
        assert "outside the 320x240 canvas" in payload["error"]

    def test_off_canvas_check_uses_the_requested_resolution(
        self, capsys, make_project, simple_image
    ):
        """The bound is the canvas actually being rendered, not a fixed number."""
        root = make_project({"image.glsl": simple_image})
        messages = []
        for resolution in ("320x240", "100x50"):
            code, payload, _ = run(
                capsys, "render", "-C", str(root), "-r", resolution, "--json",
                "--input", '[{"frame": 0, "op": "mouse_down", "pos": [400, 10]}]',
            )
            assert code == EXIT_USAGE
            messages.append(payload["error"])
        assert "0..320" in messages[0]
        assert "0..100" in messages[1]

    def test_unknown_pass(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-p", "buffer_z", "--json"
        )
        assert code == EXIT_USAGE
        assert "unknown pass" in payload["error"]


class TestInit:
    def test_list_templates(self, capsys):
        code, payload, _ = run(capsys, "init", "--list", "--json")
        assert code == EXIT_OK
        assert "basic" in payload["templates"]

    def test_scaffold_then_check(self, capsys, tmp_path):
        code, payload, _ = run(
            capsys, "init", "-C", str(tmp_path), "-t", "feedback", "--json"
        )
        assert code == EXIT_OK
        assert (tmp_path / "image.glsl").is_file()
        assert (tmp_path / "buffer_a.glsl").is_file()
        assert (tmp_path / "shadertoy.json").is_file()

    def test_refuses_to_clobber(self, capsys, tmp_path):
        run(capsys, "init", "-C", str(tmp_path), "--json")
        code, payload, _ = run(capsys, "init", "-C", str(tmp_path), "--json")
        assert code == EXIT_USAGE
        assert "refusing to overwrite" in payload["error"]

    def test_force_overwrites(self, capsys, tmp_path):
        run(capsys, "init", "-C", str(tmp_path), "--json")
        code, _, _ = run(capsys, "init", "-C", str(tmp_path), "--force", "--json")
        assert code == EXIT_OK

    def test_unknown_template(self, capsys, tmp_path):
        code, payload, _ = run(
            capsys, "init", "-C", str(tmp_path), "-t", "nope", "--json"
        )
        assert code == EXIT_USAGE
        assert "unknown template" in payload["error"]


@pytest.mark.gpu
class TestCheck:
    def test_valid_project(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_OK
        assert payload["ok"] is True
        assert payload["errors"] == 0

    def test_broken_project_reports_line(self, capsys, make_project):
        root = make_project({"image.glsl": BROKEN})
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_FAILED
        assert payload["ok"] is False
        assert payload["errors"] >= 1
        diag = payload["diagnostics"][0]
        assert diag["file"] == "image.glsl"
        assert diag["line"] == 1

    def test_human_output_goes_to_stderr(self, capsys, make_project):
        """--json output must never be polluted by prose."""
        root = make_project({"image.glsl": BROKEN})
        code, payload, err = run(capsys, "check", "-C", str(root), "--json")
        assert payload is not None, "stdout must be valid JSON"
        assert "image.glsl" in err


class TestTimingSummary:
    """The reported cost must be per-frame and must not hide warm-up work."""

    def test_single_frame_is_a_bare_number(self):
        timing = RunTiming(captured_ms=[0.705])
        assert _timing_lines(timing) == ["  gpu time 0.705 ms"]

    def test_many_frames_lead_with_the_per_frame_cost(self):
        timing = RunTiming(captured_ms=[1.0, 2.0, 3.0])
        (line,) = _timing_lines(timing)
        # 2.000 ms/frame says something about the shader; 6.000 ms only says
        # "three frames were asked for".
        assert line.startswith("  gpu time 2.000 ms/frame")
        assert "min 1.000, max 3.000" in line
        assert "x 3 = 6.000 ms" in line

    def test_warmup_frames_are_reported_not_dropped(self):
        timing = RunTiming(captured_ms=[1.0, 3.0], warmup_ms=[5.0] * 10)
        head, tail = _timing_lines(timing)
        assert "x 2 = 4.000 ms" in head
        assert "plus 10 uncaptured frame(s) 50.000 ms" in tail
        assert "54.000 ms for all 12 rendered" in tail

    def test_nothing_rendered_says_nothing(self):
        assert _timing_lines(RunTiming()) == []

    def test_totals_cover_every_rendered_frame(self):
        timing = RunTiming(captured_ms=[1.0], warmup_ms=[2.0, 4.0])
        assert timing.frames == 3
        assert timing.total_ms == 7.0
        # The mean describes captured frames only: warm-up frames of a feedback
        # shader are not the thing being measured.
        assert timing.mean_ms == 1.0
        assert timing.to_dict()["frames_rendered"] == 3


@pytest.mark.gpu
class TestRender:
    def test_in_canvas_mouse_position_renders(
        self, capsys, make_project, simple_image
    ):
        """The counterpart to the rejection: a position that fits is untouched."""
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "320x240", "--no-write", "--json",
            "--input", '[{"frame": 0, "op": "mouse_down", "pos": [319, 239]}]',
        )
        assert code == EXIT_OK
        assert payload["settings"]["inputs"]["events"][0]["pos"] == [319.0, 239.0]

    def test_writes_a_png(self, capsys, make_project, simple_image, tmp_path):
        root = make_project({"image.glsl": simple_image})
        out = tmp_path / "renders"
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "32x32",
            "-o", str(out), "--json",
        )
        assert code == EXIT_OK
        written = payload["frames"][0]["passes"]["image"]["file"]
        assert written.endswith(".png")
        assert (out / "image.png").is_file()

    def test_no_write_skips_files(self, capsys, make_project, simple_image, tmp_path):
        root = make_project({"image.glsl": simple_image})
        out = tmp_path / "renders"
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-o", str(out),
            "--no-write", "--json",
        )
        assert code == EXIT_OK
        assert "file" not in payload["frames"][0]["passes"]["image"]
        assert not out.exists()

    def test_count_and_every(self, capsys, make_project, simple_image, tmp_path):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--count", "3", "--every", "2", "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [0, 2, 4]

    def test_count_starts_at_frame(self, capsys, make_project, simple_image, tmp_path):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16", "--frame", "10",
            "--count", "3", "--every", "5", "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [10, 15, 20]

    def test_count_defaults_to_stride_one(
        self, capsys, make_project, simple_image, tmp_path
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--count", "4", "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [0, 1, 2, 3]

    def test_no_count_captures_one_frame(
        self, capsys, make_project, simple_image, tmp_path
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16", "--frame", "7",
            "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [7]

    def test_every_without_count_is_rejected(
        self, capsys, make_project, simple_image
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "--every", "5", "--no-write", "--json"
        )
        assert code == EXIT_USAGE
        assert "only applies together with --count" in payload["error"]

    def test_zero_count_is_rejected(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "--count", "0", "--no-write", "--json"
        )
        assert code == EXIT_USAGE
        assert "--count must be >= 1" in payload["error"]

    def test_all_passes(self, capsys, make_project, simple_image, tmp_path):
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image}
        )
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "-p", "all", "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert set(payload["frames"][0]["passes"]) == {"image", "buffer_a"}

    def test_broken_shader_fails(self, capsys, make_project):
        root = make_project({"image.glsl": BROKEN})
        code, payload, _ = run(capsys, "render", "-C", str(root), "--json")
        assert code == EXIT_FAILED
        assert payload["diagnostics"][0]["line"] == 1

    def test_probe_failure_sets_exit_code(
        self, capsys, make_project, halves_image, tmp_path
    ):
        root = make_project({"image.glsl": halves_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "32x32", "--no-write",
            "--probe", "4,16=0,1,0", "--json",
        )
        assert code == EXIT_FAILED
        assert payload["probe_failures"] == 1


@pytest.mark.gpu
class TestProbe:
    def test_exact_values(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "-r", "100x100",
            "--at", "10,50=1,0,0", "--at", "90,50=0,0,1", "--json",
        )
        assert code == EXIT_OK
        assert all(r["passed"] for r in payload["passes"]["image"])

    def test_wrong_expectation_fails(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "-r", "100x100",
            "--at", "10,50=0,1,0", "--json",
        )
        assert code == EXIT_FAILED
        assert payload["failures"] == 1

    def test_normalized_coordinates(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "-r", "100x100",
            "--at", "n:0.25,0.5=1,0,0", "--json",
        )
        assert code == EXIT_OK

    def test_requires_at_least_one_probe(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, _, _ = run(capsys, "probe", "-C", str(root), "--json")
        assert code == EXIT_USAGE


@pytest.mark.gpu
class TestStats:
    def test_reports_statistics(self, capsys, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "stats", "-C", str(root), "-r", "32x32", "--json"
        )
        assert code == EXIT_OK
        assert payload["passes"]["image"]["pixels"] == 1024

    def test_assert_finite_catches_nan(self, capsys, make_project):
        root = make_project({"image.glsl": NAN_SHADER})
        code, payload, _ = run(
            capsys, "stats", "-C", str(root), "-r", "16x16",
            "--assert-finite", "--json",
        )
        assert code == EXIT_FAILED
        assert payload["passes"]["image"]["has_nan"]

    def test_assert_not_black_catches_black(self, capsys, make_project):
        root = make_project({"image.glsl": BLACK})
        code, payload, _ = run(
            capsys, "stats", "-C", str(root), "-r", "16x16",
            "--assert-not-black", "--json",
        )
        assert code == EXIT_FAILED
        assert payload["problems"]

    def test_assert_not_uniform_catches_flat(self, capsys, make_project):
        root = make_project({"image.glsl": BLACK})
        code, _, _ = run(
            capsys, "stats", "-C", str(root), "-r", "16x16",
            "--assert-not-uniform", "--json",
        )
        assert code == EXIT_FAILED

    def test_assertions_pass_on_good_shader(
        self, capsys, make_project, simple_image
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "stats", "-C", str(root), "-r", "32x32", "--assert-finite",
            "--assert-not-black", "--assert-not-uniform",
            "--min-unique-colors", "8", "--json",
        )
        assert code == EXIT_OK
        assert payload["problems"] == []

    def test_histogram_included_on_request(
        self, capsys, make_project, simple_image
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "stats", "-C", str(root), "-r", "16x16",
            "--histogram", "--bins", "4", "--json",
        )
        assert code == EXIT_OK
        assert len(payload["passes"]["image"]["histogram"]["r"]) == 4


@pytest.mark.gpu
class TestGoldenWorkflow:
    def test_bless_then_test_passes(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        code, _, _ = run(capsys, "bless", "-C", str(root), "-r", "32x32", "--json")
        assert code == EXIT_OK
        assert (root / "golden" / "image_f0000.png").is_file()

        code, payload, _ = run(capsys, "test", "-C", str(root), "-r", "32x32", "--json")
        assert code == EXIT_OK
        assert payload["failures"] == 0

    def test_missing_golden_fails(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        code, payload, _ = run(capsys, "test", "-C", str(root), "-r", "32x32", "--json")
        assert code == EXIT_FAILED
        assert payload["results"][0]["status"] == "missing"

    def test_regression_is_caught(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        run(capsys, "bless", "-C", str(root), "-r", "32x32", "--json")
        # Change the shader: blue channel drifts noticeably.
        (root / "image.glsl").write_text(
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = f.x < iResolution.x*0.5 ? vec4(1,0,0,1) : vec4(0,0.5,1,1);\n"
            "}\n"
        )
        code, payload, _ = run(capsys, "test", "-C", str(root), "-r", "32x32", "--json")
        assert code == EXIT_FAILED
        assert payload["results"][0]["status"] == "fail"
        assert payload["results"][0]["max_diff"] > 2

    def test_size_mismatch_reported(self, capsys, make_project, halves_image):
        root = make_project({"image.glsl": halves_image})
        run(capsys, "bless", "-C", str(root), "-r", "32x32", "--json")
        code, payload, _ = run(capsys, "test", "-C", str(root), "-r", "64x64", "--json")
        assert code == EXIT_FAILED
        assert payload["results"][0]["status"] == "size-mismatch"


@pytest.mark.gpu
class TestInfo:
    def test_reports_context_and_devices(self, capsys):
        code, payload, _ = run(capsys, "info", "--json")
        assert code == EXIT_OK
        assert payload["context"]["gl_renderer"]
        assert isinstance(payload["devices"], list)

    def test_includes_project_when_present(
        self, capsys, make_project, simple_image
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(capsys, "info", "-C", str(root), "--json")
        assert code == EXIT_OK
        assert payload["project"]["passes"]["image"]


@pytest.mark.gpu
class TestCommonPortability:
    UNSAFE_COMMON = "vec3 pal() { return vec3(iTime); }\n"
    SAFE_COMMON = "vec4 when() { return iDate; }\n"
    IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(pal(), 1.0); }\n"
    SAFE_IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = when(); }\n"

    def test_warns_by_default(self, capsys, make_project):
        root = make_project(
            {"common.glsl": self.UNSAFE_COMMON, "image.glsl": self.IMAGE}
        )
        code, payload, err = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_OK, "portability issues are warnings, not errors"
        assert len(payload["portability"]) == 1
        assert payload["portability"][0]["file"] == "common.glsl"
        assert "iTime" in err

    def test_can_be_disabled(self, capsys, make_project):
        root = make_project(
            {"common.glsl": self.UNSAFE_COMMON, "image.glsl": self.IMAGE}
        )
        code, payload, _ = run(
            capsys, "check", "-C", str(root), "--no-portability", "--json"
        )
        assert code == EXIT_OK
        assert payload["portability"] == []

    def test_clean_common_produces_no_warnings(self, capsys, make_project):
        root = make_project(
            {"common.glsl": self.SAFE_COMMON, "image.glsl": self.SAFE_IMAGE}
        )
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_OK
        assert payload["portability"] == []

    def test_self_declaration_workaround_is_a_real_error(self, capsys, make_project):
        """Declaring `uniform float iTime;` in Common collides with the prelude,
        exactly as it collides with the site's own pass header."""
        root = make_project(
            {
                "common.glsl": "uniform float iTime;\n" + self.UNSAFE_COMMON,
                "image.glsl": self.IMAGE,
            }
        )
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_FAILED
        errors = [d for d in payload["diagnostics"] if d["severity"] == "error"]
        assert errors and errors[0]["file"] == "common.glsl"
        assert errors[0]["line"] == 1
        # The prior declaration must not leak a raw composed line number.
        assert "0(" not in errors[0]["message"]


class TestDeviceSelection:
    """The CLI surface for choosing a renderer. `--allow-software` is routinely
    mistaken for "render on CPU"; it only permits CPU as a fallback."""

    def _renderer(self, capsys, root, *extra):
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--no-write", "--json", *extra,
        )
        return code, payload

    @pytest.mark.gpu
    def test_defaults_to_hardware(
        self, capsys, make_project, simple_image, auto_device_only
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload = self._renderer(capsys, root)
        assert code == EXIT_OK
        assert payload["renderer"]["software"] is False

    @pytest.mark.gpu
    def test_allow_software_does_not_override_hardware(
        self, capsys, make_project, simple_image, auto_device_only
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload = self._renderer(capsys, root, "--allow-software")
        assert code == EXIT_OK
        assert payload["renderer"]["software"] is False, (
            "--allow-software must not take precedence over an available GPU"
        )

    @pytest.mark.cpu
    def test_explicit_device_selects_software(
        self, capsys, make_project, simple_image
    ):
        from shadertoy_local.context import enumerate_devices

        software = [d for d in enumerate_devices() if d.is_software]
        if not software:
            pytest.skip("no software EGL device available")
        root = make_project({"image.glsl": simple_image})
        code, payload = self._renderer(capsys, root, "--device", str(software[0].index))
        assert code == EXIT_OK
        assert payload["renderer"]["software"] is True
        assert payload["renderer"]["device_index"] == software[0].index

    @pytest.mark.gpu
    def test_nonexistent_device_is_a_clear_error(
        self, capsys, make_project, simple_image
    ):
        root = make_project({"image.glsl": simple_image})
        code, payload = self._renderer(capsys, root, "--device", "99")
        assert code == EXIT_ENVIRONMENT
        assert "does not exist" in payload["error"]

    @pytest.mark.gpu
    def test_glx_backend_fails_without_a_display(
        self, capsys, make_project, simple_image
    ):
        """GLX needs an X server; the error must say so rather than be cryptic."""
        import os

        if os.environ.get("DISPLAY"):
            pytest.skip("a display is available, so GLX may succeed")
        root = make_project({"image.glsl": simple_image})
        code, payload = self._renderer(capsys, root, "--backend", "glx")
        assert code == EXIT_ENVIRONMENT
        assert "error" in payload


@pytest.mark.gpu
class TestStructTernaryFailsHard:
    """A struct ternary compiles locally but not on shadertoy.com, so `check`
    must fail rather than merely warn."""

    SOURCE = (
        "struct Ray { vec3 o; vec3 d; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Ray a = Ray(vec3(0.0), vec3(1.0));\n"
        "    Ray b = Ray(vec3(1.0), vec3(0.0));\n"
        "    Ray r = fragCoord.x > 1.0 ? a : b;\n"
        "    fragColor = vec4(r.o, 1.0);\n"
        "}\n"
    )
    PORTABLE = (
        "struct Ray { vec3 o; vec3 d; };\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    Ray a = Ray(vec3(0.0), vec3(1.0));\n"
        "    Ray b = Ray(vec3(1.0), vec3(0.0));\n"
        "    Ray r = b;\n"
        "    if (fragCoord.x > 1.0) { r = a; }\n"
        "    fragColor = vec4(r.o, 1.0);\n"
        "}\n"
    )

    def test_check_fails_with_exit_1(self, capsys, make_project):
        root = make_project({"image.glsl": self.SOURCE})
        code, payload, err = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_FAILED
        assert payload["ok"] is False
        assert payload["errors"] >= 1
        entry = payload["portability"][0]
        assert entry["severity"] == "error"
        assert entry["code"] == "ST-TERNARY"
        assert entry["file"] == "image.glsl"
        assert entry["line"] == 5
        assert "ST-TERNARY" in err

    def test_it_still_compiles_locally(self, capsys, make_project):
        """Proof the failure is a portability judgement, not a compile error:
        the desktop driver accepts this happily."""
        root = make_project({"image.glsl": self.SOURCE})
        code, payload, _ = run(
            capsys, "check", "-C", str(root), "--no-portability", "--json"
        )
        assert code == EXIT_OK
        assert payload["errors"] == 0

    def test_portable_rewrite_passes(self, capsys, make_project):
        root = make_project({"image.glsl": self.PORTABLE})
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_OK
        assert payload["portability"] == []

    def test_common_tab_warning_does_not_fail_the_command(
        self, capsys, make_project
    ):
        """Contrast: the Common-tab finding is cosmetic on the site, so it warns
        without failing."""
        root = make_project(
            {
                "common.glsl": "vec3 pal(){ return vec3(iTime); }\n",
                "image.glsl": "void mainImage(out vec4 c, in vec2 f){ c=vec4(pal(),1.0); }\n",
            }
        )
        code, payload, _ = run(capsys, "check", "-C", str(root), "--json")
        assert code == EXIT_OK
        assert payload["warnings"] >= 1
        assert payload["errors"] == 0


@pytest.mark.gpu
class TestInfoRuntimeCheck:
    """`info` compiles and runs a shader on every device by default."""

    def test_reports_runtime_per_device(self, capsys):
        code, payload, _ = run(capsys, "info", "--json")
        assert code == EXIT_OK
        assert payload["devices"], "expected at least one device"
        for dev in payload["devices"]:
            runtime = dev["runtime"]
            assert runtime is not None, f"device {dev['index']} has no runtime result"
            assert runtime["ok"] is True, f"device {dev['index']}: {runtime['error']}"
            assert runtime["stage"] == "ok"
            assert runtime["max_error"] is not None
            assert runtime["shader_ms"] > 0
            assert runtime["context_ms"] > 0

    def test_software_devices_are_checked_too(self, capsys):
        code, payload, _ = run(capsys, "info", "--json")
        assert code == EXIT_OK
        # Every enumerated device gets a verdict, software included.
        assert all(d["runtime"] is not None for d in payload["devices"])

    def test_human_output_mentions_runtime(self, capsys):
        code, _, err = run(capsys, "info")
        assert code == EXIT_OK
        assert "runtime:" in err
        assert "shader" in err

    def test_can_be_skipped(self, capsys):
        code, payload, err = run(capsys, "info", "--no-runtime-check", "--json")
        assert code == EXIT_OK
        assert all("runtime" not in d for d in payload["devices"])
        assert "runtime:" not in err

    def test_failing_device_is_surfaced(self, capsys, monkeypatch):
        """A device that cannot run a shader must be reported as failing rather
        than silently listed as present."""
        from shadertoy_local.context import enumerate_devices
        from shadertoy_local.selftest import RuntimeCheck

        real = enumerate_devices()
        broken = [
            RuntimeCheck(
                device_index=d.index,
                ok=False,
                stage="compile",
                error="synthetic failure",
            )
            for d in real
        ]
        monkeypatch.setattr(
            "shadertoy_local.selftest.check_all_devices", lambda **kw: broken
        )
        code, payload, err = run(capsys, "info", "--json")
        assert code == EXIT_ENVIRONMENT
        assert payload["ok"] is False
        assert "synthetic failure" in err
        assert "no device could compile and run a shader" in err

    def test_partial_failure_still_succeeds(self, capsys, monkeypatch):
        """One working device is enough for rendering to be possible."""
        from shadertoy_local.context import enumerate_devices
        from shadertoy_local.selftest import RuntimeCheck

        real = enumerate_devices()
        if len(real) < 2:
            pytest.skip("needs at least two devices")
        mixed = [
            RuntimeCheck(
                device_index=real[0].index, ok=True, stage="ok",
                gl_version="4.6", context_ms=1.0, shader_ms=1.0, max_error=0.0,
            ),
            RuntimeCheck(
                device_index=real[1].index, ok=False, stage="render",
                error="synthetic failure",
            ),
        ]
        monkeypatch.setattr(
            "shadertoy_local.selftest.check_all_devices", lambda **kw: mixed
        )
        code, payload, _ = run(capsys, "info", "--json")
        assert code == EXIT_OK
        assert payload["ok"] is True


@pytest.mark.gpu
class TestPrechargeFlag:
    ACCUMULATOR = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = texture(iChannel0, f/iResolution.xy) + vec4(1.0);\n"
        "}\n"
    )
    PASSTHROUGH = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = texture(iChannel0, f/iResolution.xy) / 1000.0;\n"
        "}\n"
    )

    def _root(self, make_project):
        return make_project(
            {"buffer_a.glsl": self.ACCUMULATOR, "image.glsl": self.PASSTHROUGH},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )

    def _frames_rendered(self, capsys, root, *extra):
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "-r", "16x16",
            "--at", "1,1", "--json", *extra,
        )
        assert code == EXIT_OK
        return round(payload["passes"]["image"][0]["rgba"][0] * 1000)

    def test_default_warms_up_fully(self, capsys, make_project):
        root = self._root(make_project)
        assert self._frames_rendered(capsys, root, "--frame", "30") == 31

    def test_explicit_window(self, capsys, make_project):
        root = self._root(make_project)
        assert self._frames_rendered(
            capsys, root, "--frame", "30", "--precharge", "5"
        ) == 6

    def test_all_keyword(self, capsys, make_project):
        root = self._root(make_project)
        assert self._frames_rendered(
            capsys, root, "--frame", "30", "--precharge", "all"
        ) == 31

    def test_zero(self, capsys, make_project):
        root = self._root(make_project)
        assert self._frames_rendered(
            capsys, root, "--frame", "30", "--precharge", "0"
        ) == 1

    def test_negative_rejected(self, capsys, make_project):
        root = self._root(make_project)
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "--at", "0,0",
            "--precharge", "-3", "--json",
        )
        assert code == EXIT_USAGE
        assert ">= 0" in payload["error"]

    def test_garbage_rejected(self, capsys, make_project):
        root = self._root(make_project)
        code, payload, _ = run(
            capsys, "probe", "-C", str(root), "--at", "0,0",
            "--precharge", "banana", "--json",
        )
        assert code == EXIT_USAGE
        assert "frame count or 'all'" in payload["error"]

    def test_precharge_applies_to_the_first_capture_only(
        self, capsys, make_project, tmp_path
    ):
        """With --count, warm-up precedes the first frame; later captures keep
        accumulating contiguously."""
        root = self._root(make_project)
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--frame", "20", "--count", "3", "--every", "5",
            "--precharge", "5", "--no-write", "--stats", "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [20, 25, 30]
        # Warm-up starts at 15, so frame 20 is the 6th rendered frame.
        maxima = [
            f["passes"]["image"]["stats"]["channels"]["r"]["max"] * 1000
            for f in payload["frames"]
        ]
        assert [round(v) for v in maxima] == [6, 11, 16]

    def test_no_simulate_flags_are_gone(self, capsys, make_project):
        """--simulate/--no-simulate were three flags for one knob; argparse
        rejects them outright now."""
        root = self._root(make_project)
        with pytest.raises(SystemExit) as excinfo:
            main(["render", "-C", str(root), "--no-simulate"])
        assert excinfo.value.code == EXIT_USAGE
        assert "unrecognized arguments" in capsys.readouterr().err
