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
    main,
)

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
            capsys, "render", "-C", str(root), "--key", "nonsense", "--json"
        )
        assert code == EXIT_USAGE
        assert "unknown key" in payload["error"]

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


@pytest.mark.gpu
class TestRender:
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

    def test_frame_range_spec(self, capsys, make_project, simple_image, tmp_path):
        root = make_project({"image.glsl": simple_image})
        code, payload, _ = run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--frames", "0-4:2", "-o", str(tmp_path / "o"), "--json",
        )
        assert code == EXIT_OK
        assert [f["frame"] for f in payload["frames"]] == [0, 2, 4]

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
