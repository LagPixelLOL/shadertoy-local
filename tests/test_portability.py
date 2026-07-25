"""Common-tab portability linting.

Encodes the observed shadertoy.com behaviour: the Common tab is validated
standalone against a minimal header in which only iDate and iSampleRate exist.
"""

from __future__ import annotations

from shadertoy_local.portability import (
    COMMON_SAFE_UNIFORMS,
    PASS_SPECIFIC_UNIFORMS,
    lint_common,
    strip_comments,
)
from shadertoy_local.project import load_project


def _lint(make_project, common: str, image: str | None = None):
    image = image or "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"
    root = make_project({"common.glsl": common, "image.glsl": image})
    return lint_common(load_project(root))


class TestStripComments:
    def test_line_numbering_is_preserved(self):
        source = "a\n// comment\nb\n"
        assert len(strip_comments(source).split("\n")) == len(source.split("\n"))

    def test_block_comment_newlines_survive(self):
        source = "a\n/* one\ntwo\nthree */\nb\n"
        cleaned = strip_comments(source)
        assert len(cleaned.split("\n")) == len(source.split("\n"))
        assert "one" not in cleaned and "three" not in cleaned

    def test_column_positions_are_preserved(self):
        source = "int x; // iTime\n"
        cleaned = strip_comments(source)
        assert len(cleaned.split("\n")[0]) == len(source.split("\n")[0])
        assert cleaned.startswith("int x;")

    def test_code_is_untouched(self):
        source = "vec3 f() { return vec3(1.0); }\n"
        assert strip_comments(source) == source


class TestLint:
    def test_flags_pass_specific_uniform(self, make_project):
        diagnostics = _lint(make_project, "float t() { return iTime; }\n")
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert diag.severity == "warning"
        assert diag.file == "common.glsl"
        assert diag.line == 1
        assert diag.code == "ST-COMMON"
        assert "iTime" in diag.message

    def test_reports_accurate_column(self, make_project):
        source = "float t() { return iTime; }\n"
        (diag,) = _lint(make_project, source)
        assert source[diag.column - 1 :].startswith("iTime")

    def test_safe_uniforms_are_silent(self, make_project):
        source = "vec4 d() { return iDate * iSampleRate; }\n"
        assert _lint(make_project, source) == []

    def test_safe_set_and_flagged_set_are_disjoint(self):
        assert COMMON_SAFE_UNIFORMS.isdisjoint(PASS_SPECIFIC_UNIFORMS)

    def test_all_pass_specific_uniforms_are_detected(self, make_project):
        for name in PASS_SPECIFIC_UNIFORMS:
            source = f"// probe\nfloat f() {{ return float({name}.x); }}\n"
            diagnostics = _lint(make_project, source)
            assert any(name in d.message for d in diagnostics), name

    def test_comments_are_ignored(self, make_project):
        source = (
            "// iTime here is only a comment\n"
            "/* and iResolution\n   and iMouse */\n"
            "float f() { return 1.0; }\n"
        )
        assert _lint(make_project, source) == []

    def test_define_body_is_exempt(self, make_project):
        """An unexpanded macro body never reaches the compiler, so the site
        does not flag it -- which is what makes the struct pattern usable."""
        source = "#define CAPTURE ST(iResolution, iTime)\nfloat f(){ return 1.0; }\n"
        assert _lint(make_project, source) == []

    def test_multiline_define_is_exempt(self, make_project):
        source = (
            "#define CAPTURE ST( \\\n"
            "    iResolution, \\\n"
            "    iTime)\n"
            "float f(){ return 1.0; }\n"
        )
        assert _lint(make_project, source) == []

    def test_code_after_multiline_define_is_still_linted(self, make_project):
        source = (
            "#define A ST( \\\n"
            "    iResolution)\n"
            "float f(){ return iTime; }\n"
        )
        diagnostics = _lint(make_project, source)
        assert len(diagnostics) == 1
        assert diagnostics[0].line == 3

    def test_substring_names_are_not_matched(self, make_project):
        source = "float myiTimeValue = 1.0;\nfloat iTimeless = 2.0;\n"
        assert _lint(make_project, source) == []

    def test_one_warning_per_uniform_per_line(self, make_project):
        source = "float f(){ return iTime + iTime + iResolution.x; }\n"
        diagnostics = _lint(make_project, source)
        assert {d.message.split()[0] for d in diagnostics} == {"iTime", "iResolution"}
        assert len(diagnostics) == 2

    def test_reports_each_offending_line(self, make_project):
        source = "float a(){ return iTime; }\nfloat b(){ return iTime; }\n"
        assert [d.line for d in _lint(make_project, source)] == [1, 2]

    def test_no_common_file_means_no_warnings(self, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        assert lint_common(load_project(root)) == []

    def test_image_pass_uniforms_are_never_linted(self, make_project):
        """Only Common is restricted; passes may use anything."""
        image = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(iTime, iMouse.x, iResolution.x, float(iFrame));\n"
            "}\n"
        )
        assert _lint(make_project, "float f(){ return 1.0; }\n", image) == []

    def test_diagnostic_is_serialisable(self, make_project):
        import json

        (diag,) = _lint(make_project, "float t(){ return iTime; }\n")
        assert json.loads(json.dumps(diag.to_dict()))["code"] == "ST-COMMON"


class TestShippedExamples:
    def test_portable_example_is_clean(self):
        """examples/06 is the reference for the recommended pattern."""
        from .conftest import EXAMPLES_DIR

        project = load_project(EXAMPLES_DIR / "06-portable-common")
        assert lint_common(project) == []

    def test_every_example_is_clean(self):
        from .conftest import EXAMPLES_DIR

        for path in sorted(EXAMPLES_DIR.iterdir()):
            if not path.is_dir():
                continue
            warnings = lint_common(load_project(path))
            assert warnings == [], f"{path.name}: {[w.message for w in warnings]}"
