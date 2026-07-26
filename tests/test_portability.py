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


def _ternary(make_project, common: str = "", image: str = ""):
    from shadertoy_local.portability import lint_struct_ternary

    files = {"image.glsl": image or "void mainImage(out vec4 c, in vec2 f){ c=vec4(1.0); }\n"}
    if common:
        files["common.glsl"] = common
    return lint_struct_ternary(load_project(make_project(files)))


class TestStructTernary:
    """shadertoy.com runs WebGL, where ?: on a struct fails to COMPILE. Unlike
    the Common-tab check (cosmetic there), this must be an error."""

    STRUCT = "struct Ray { vec3 o; vec3 d; };\n"

    def test_is_an_error_not_a_warning(self, make_project):
        image = (
            self.STRUCT
            + "void mainImage(out vec4 c, in vec2 f){\n"
            "    Ray a = Ray(vec3(0.0), vec3(1.0));\n"
            "    Ray b = Ray(vec3(1.0), vec3(0.0));\n"
            "    Ray r = f.x > 1.0 ? a : b;\n"
            "    c = vec4(r.o, 1.0);\n"
            "}\n"
        )
        (diag,) = _ternary(make_project, image=image)
        assert diag.severity == "error"
        assert diag.is_error
        assert diag.code == "ST-TERNARY"
        assert "Ray" in diag.message

    def test_detects_struct_typed_declaration(self, make_project):
        image = (
            self.STRUCT
            + "void mainImage(out vec4 c, in vec2 f){\n"
            "    Ray a, b;\n"
            "    Ray r = f.x > 1.0 ? a : b;\n"
            "    c = vec4(r.o, 1.0);\n"
            "}\n"
        )
        assert len(_ternary(make_project, image=image)) == 1

    def test_detects_constructor_branches(self, make_project):
        image = (
            self.STRUCT
            + "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4((f.x > 1.0 ? Ray(vec3(0.0), vec3(1.0))\n"
            "                        : Ray(vec3(1.0), vec3(0.0))).o, 1.0);\n"
            "}\n"
        )
        assert len(_ternary(make_project, image=image)) >= 1

    def test_detects_return_of_bare_struct_variables(self, make_project):
        """The usual shape, and the one needing whole-source variable types:
        a and b are declared in the parameter list, not the return statement."""
        common = self.STRUCT + "Ray pick(bool t, Ray a, Ray b){ return t ? a : b; }\n"
        diagnostics = _ternary(make_project, common=common)
        assert len(diagnostics) == 1
        assert diagnostics[0].file == "common.glsl"
        assert diagnostics[0].line == 2

    def test_struct_declared_in_common_used_in_image(self, make_project):
        common = self.STRUCT
        image = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    Ray a, b;\n"
            "    Ray r = f.x > 1.0 ? a : b;\n"
            "    c = vec4(r.o, 1.0);\n"
            "}\n"
        )
        diagnostics = _ternary(make_project, common=common, image=image)
        assert len(diagnostics) == 1
        assert diagnostics[0].file == "image.glsl"

    # -- false positives ------------------------------------------------

    def test_member_access_result_is_not_flagged(self, make_project):
        """`t ? a.o : b.o` evaluates to vec3, which WebGL accepts."""
        image = (
            self.STRUCT
            + "void mainImage(out vec4 c, in vec2 f){\n"
            "    Ray a, b;\n"
            "    vec3 v = f.x > 1.0 ? a.o : b.o;\n"
            "    c = vec4(v, 1.0);\n"
            "}\n"
        )
        assert _ternary(make_project, image=image) == []

    def test_scalar_ternary_is_not_flagged(self, make_project):
        image = (
            self.STRUCT
            + "void mainImage(out vec4 c, in vec2 f){\n"
            "    float x = f.x > 1.0 ? 1.0 : 0.0;\n"
            "    c = vec4(x);\n"
            "}\n"
        )
        assert _ternary(make_project, image=image) == []

    def test_function_name_is_not_treated_as_a_variable(self, make_project):
        """`Ray pick(...)` declares a function; treating `pick` as struct-typed
        would misfire on any ternary mentioning it."""
        common = (
            self.STRUCT
            + "Ray build(float v){ Ray r; r.o = vec3(v); r.d = vec3(0.0); return r; }\n"
            "float scale(bool t){ return t ? 1.0 : 2.0; }\n"
        )
        assert _ternary(make_project, common=common) == []

    def test_no_structs_means_no_work(self, make_project):
        image = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = f.x > 1.0 ? vec4(1.0) : vec4(0.0);\n"
            "}\n"
        )
        assert _ternary(make_project, image=image) == []

    def test_commented_out_ternary_is_ignored(self, make_project):
        common = self.STRUCT + "// Ray r = t ? a : b;\n"
        assert _ternary(make_project, common=common) == []

    def test_shipped_examples_are_clean(self):
        from shadertoy_local.portability import lint_struct_ternary
        from .conftest import EXAMPLES_DIR

        for path in sorted(EXAMPLES_DIR.iterdir()):
            if not path.is_dir():
                continue
            found = lint_struct_ternary(load_project(path))
            assert found == [], f"{path.name}: {[d.message for d in found]}"


class TestSeverityDistinction:
    """The two checks must not be conflated: one predicts cosmetic editor noise,
    the other predicts a hard compile failure."""

    def test_common_tab_stays_a_warning(self, make_project):
        common = "float t(){ return iTime; }\n"
        diagnostics = _lint(make_project, common)
        assert all(d.severity == "warning" for d in diagnostics)
        assert all(not d.is_error for d in diagnostics)

    def test_lint_all_reports_every_check(self, make_project):
        from shadertoy_local.portability import lint_all

        common = (
            "struct Ray { vec3 o; };\n"
            "float t(){ return iTime; }\n"
            "Ray pick(bool f, Ray a, Ray b){ return f ? a : b; }\n"
        )
        root = make_project(
            {
                "common.glsl": common,
                "image.glsl": (
                    "void mainImage(out vec4 c, in vec2 f){"
                    " float active = t(); c=vec4(active); }\n"
                ),
            }
        )
        codes = {d.code: d.severity for d in lint_all(load_project(root))}
        assert codes == {
            "ST-COMMON": "warning",
            "ST-TERNARY": "error",
            "ST-RESERVED": "error",
        }


def _reserved(make_project, image: str, common: str = ""):
    from shadertoy_local.portability import lint_reserved_words

    files = {"image.glsl": image}
    if common:
        files["common.glsl"] = common
    return lint_reserved_words(load_project(make_project(files)))


class TestReservedWords:
    """GLSL ES reserves words that desktop drivers accept as identifiers
    without a murmur -- `float active;` compiles on NVIDIA GL 4.6 and stops
    the same shader dead on shadertoy.com. A hard failure there must be a
    hard failure here; found by pasting a shader that had passed every local
    check.
    """

    def test_flags_reserved_identifier_as_error(self, make_project):
        image = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    float active = 1.0;\n"
            "    c = vec4(active);\n"
            "}\n"
        )
        diagnostics = _reserved(make_project, image)
        assert len(diagnostics) == 2  # declaration and use
        assert all(d.severity == "error" for d in diagnostics)
        assert all(d.is_error for d in diagnostics)
        assert all(d.code == "ST-RESERVED" for d in diagnostics)
        assert "active" in diagnostics[0].message

    def test_reports_accurate_position(self, make_project):
        image = "void mainImage(out vec4 c, in vec2 f){ float filter = 1.0; c = vec4(filter); }\n"
        diagnostics = _reserved(make_project, image)
        first = diagnostics[0]
        assert first.line == 1
        assert image[first.column - 1 :].startswith("filter")

    def test_flags_common_too(self, make_project):
        common = "float half = 0.5;\n"
        image = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"
        diagnostics = _reserved(make_project, image, common=common)
        assert len(diagnostics) == 1
        assert diagnostics[0].file == "common.glsl"
        assert diagnostics[0].pass_name == "common"

    def test_substrings_are_not_matched(self, make_project):
        image = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    float activeCount = 1.0;  float interactive = 2.0;\n"
            "    c = vec4(activeCount * interactive);\n"
            "}\n"
        )
        assert _reserved(make_project, image) == []

    def test_comments_are_ignored(self, make_project):
        image = (
            "// the active turret, using a filter\n"
            "/* input and output */\n"
            "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"
        )
        assert _reserved(make_project, image) == []

    def test_define_bodies_are_not_exempt(self, make_project):
        """Unlike ST-COMMON: a reserved word in a macro fails on the site
        wherever the macro is expanded, so the definition is the right place
        to point at."""
        image = (
            "#define PICK(x) float active = (x);\n"
            "void mainImage(out vec4 c, in vec2 f){ PICK(1.0) c = vec4(active); }\n"
        )
        diagnostics = _reserved(make_project, image)
        assert any(d.line == 1 for d in diagnostics)

    def test_every_reserved_word_is_detected(self, make_project):
        from shadertoy_local.portability import RESERVED_ES_WORDS

        for word in sorted(RESERVED_ES_WORDS):
            image = (
                "void mainImage(out vec4 c, in vec2 f){\n"
                f"    float {word} = 1.0;\n"
                "    c = vec4(1.0);\n"
                "}\n"
            )
            diagnostics = _reserved(make_project, image)
            assert diagnostics, f"{word} was not flagged"

    def test_clean_shader_is_silent(self, make_project):
        image = "void mainImage(out vec4 c, in vec2 f){ float growth = 1.0; c = vec4(growth); }\n"
        assert _reserved(make_project, image) == []

    def test_every_example_is_clean(self):
        from pathlib import Path

        from shadertoy_local.portability import lint_reserved_words

        examples = Path(__file__).resolve().parent.parent / "examples"
        for path in sorted(examples.iterdir()):
            if not path.is_dir():
                continue
            found = lint_reserved_words(load_project(path))
            assert found == [], f"{path.name}: {[d.message for d in found]}"
