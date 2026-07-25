"""Driver log parsing and remapping to user source positions."""

from __future__ import annotations

from shadertoy_local.compose import ComposedShader, Origin, compose_pass
from shadertoy_local.diagnostics import (
    Diagnostic,
    format_diagnostics,
    parse_log,
    summarize,
)
from shadertoy_local.project import load_project


class TestFormatParsing:
    """Each vendor spells diagnostics differently; all must parse."""

    def test_nvidia_format(self):
        log = '0(28) : error C1503: undefined variable "foo"'
        (diag,) = parse_log(log)
        assert diag.severity == "error"
        assert diag.composed_line == 28
        assert diag.code == "C1503"
        assert diag.message == 'undefined variable "foo"'

    def test_mesa_format_includes_column(self):
        log = "0:28(10): error: no matching function for call to bar()"
        (diag,) = parse_log(log)
        assert (diag.composed_line, diag.column) == (28, 10)
        assert diag.severity == "error"

    def test_amd_format(self):
        log = "ERROR: 0:28: 'foo' : undeclared identifier"
        (diag,) = parse_log(log)
        assert diag.composed_line == 28
        assert diag.severity == "error"

    def test_warnings_are_classified(self):
        log = "0(12) : warning C7011: implicit cast from int to float"
        (diag,) = parse_log(log)
        assert diag.severity == "warning"
        assert not diag.is_error

    def test_positionless_error(self):
        (diag,) = parse_log("ERROR: too many uniforms")
        assert diag.severity == "error"
        assert diag.composed_line is None

    def test_moderngl_decoration_is_stripped(self):
        log = (
            "GLSL Compiler failed\n\n"
            "fragment_shader\n"
            "===============\n"
            "0(5) : error C0000: syntax error\n"
        )
        diagnostics = parse_log(log)
        assert len(diagnostics) == 1
        assert diagnostics[0].composed_line == 5

    def test_duplicates_are_collapsed(self):
        log = "0(5) : error C0127: dup\n0(5) : error C0127: dup\n"
        assert len(parse_log(log)) == 1

    def test_unrecognised_lines_are_surfaced_not_swallowed(self):
        """Silently dropping an unparsed message would hide real failures."""
        diagnostics = parse_log("something entirely unexpected")
        assert len(diagnostics) == 1
        assert diagnostics[0].message == "something entirely unexpected"

    def test_empty_log_yields_nothing(self):
        assert parse_log("") == []
        assert parse_log("\n  \n") == []


class TestRemapping:
    def _composed(self):
        return ComposedShader(
            pass_name="image",
            source="\n".join(f"line{i}" for i in range(1, 11)),
            origins=[
                None,
                None,
                Origin("common.glsl", 1),
                Origin("common.glsl", 2),
                Origin("image.glsl", 1),
                Origin("image.glsl", 2),
                Origin("image.glsl", 3),
                None,
                None,
                None,
            ],
        )

    def test_maps_into_the_right_file(self):
        composed = self._composed()
        (diag,) = parse_log("0(6) : error C1503: bad", composed, "image")
        assert (diag.file, diag.line) == ("image.glsl", 2)
        assert diag.composed_line == 6
        assert not diag.approximate

    def test_maps_into_common(self):
        composed = self._composed()
        (diag,) = parse_log("0(4) : error C1503: bad", composed, "image")
        assert (diag.file, diag.line) == ("common.glsl", 2)

    def test_generated_line_is_approximate(self):
        composed = self._composed()
        (diag,) = parse_log("0(9) : error C1503: bad", composed, "image")
        assert diag.approximate
        # Attributed to the last real user line rather than nothing at all.
        assert (diag.file, diag.line) == ("image.glsl", 3)

    def test_location_string(self):
        composed = self._composed()
        (diag,) = parse_log("0:6(3): error: bad", composed, "image")
        assert diag.location() == "image.glsl:2:3"

    def test_pass_name_is_recorded(self):
        (diag,) = parse_log("0(6) : error C1: x", self._composed(), "buffer_a")
        assert diag.pass_name == "buffer_a"


def test_summarize_counts():
    diagnostics = [
        Diagnostic(severity="error", message="a"),
        Diagnostic(severity="error", message="b"),
        Diagnostic(severity="warning", message="c"),
    ]
    assert summarize(diagnostics) == (2, 1)


def test_format_includes_snippet_and_caret(make_project, simple_image):
    project = load_project(make_project({"image.glsl": simple_image}))
    composed = compose_pass(project, project.passes["image"])
    # Find a composed line that belongs to the user's file.
    line = next(
        i + 1 for i, o in enumerate(composed.origins) if o and o.file == "image.glsl"
    )
    diag = parse_log(f"0:{line}(5): error: boom", composed, "image")
    text = format_diagnostics(diag, composed)
    assert "image.glsl:" in text
    assert "boom" in text
    assert "^" in text  # caret is drawn when a column is known


def test_format_of_empty_list_is_empty():
    assert format_diagnostics([]) == ""


def test_diagnostic_dict_is_serialisable():
    import json

    diag = Diagnostic(
        severity="error", message="x", file="image.glsl", line=3, composed_line=30
    )
    assert json.loads(json.dumps(diag.to_dict()))["line"] == 3


class TestEmbeddedPositions:
    """Drivers sometimes put a second position inside the message text, e.g.
    NVIDIA's `conflicts with previous declaration at 0(5)`. Leaving it raw
    leaks a composed line number into otherwise remapped output."""

    def _composed(self):
        return ComposedShader(
            pass_name="image",
            source="\n".join(f"line{i}" for i in range(1, 9)),
            origins=[
                None,
                None,
                Origin("common.glsl", 1),
                Origin("common.glsl", 2),
                Origin("image.glsl", 1),
                None,
                None,
                None,
            ],
        )

    def test_embedded_position_is_remapped(self):
        log = '0(4) : error C1038: declaration of "iTime" conflicts with previous declaration at 0(3)'
        (diag,) = parse_log(log, self._composed(), "image")
        assert "common.glsl:1" in diag.message
        assert "0(3)" not in diag.message

    def test_generated_embedded_position_is_labelled(self):
        log = "0(4) : error C1038: conflicts with previous declaration at 0(2)"
        (diag,) = parse_log(log, self._composed(), "image")
        assert "<generated>:2" in diag.message

    def test_message_without_embedded_position_is_untouched(self):
        log = '0(5) : error C1503: undefined variable "foo"'
        (diag,) = parse_log(log, self._composed(), "image")
        assert diag.message == 'undefined variable "foo"'

    def test_no_composed_shader_leaves_message_alone(self):
        log = "0(4) : error C1038: conflicts with previous declaration at 0(3)"
        (diag,) = parse_log(log)
        assert "0(3)" in diag.message
