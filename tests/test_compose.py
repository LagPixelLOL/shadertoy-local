"""Shader composition and the composed-line -> source-line mapping.

The mapping is the load-bearing part: if it drifts, every reported error points
at the wrong line, which is worse than reporting nothing.
"""

from __future__ import annotations

import pytest

from shadertoy_local.compose import compose_pass, vertex_source
from shadertoy_local.project import ProjectError, load_project


def _compose(root):
    project = load_project(root)
    return project, compose_pass(project, project.passes["image"])


def test_prelude_declares_the_shadertoy_uniforms(make_project, simple_image):
    _, composed = _compose(make_project({"image.glsl": simple_image}))
    for name in (
        "iResolution",
        "iTime",
        "iTimeDelta",
        "iFrameRate",
        "iFrame",
        "iChannelTime",
        "iChannelResolution",
        "iMouse",
        "iDate",
        "iSampleRate",
        "iChannel0",
        "iChannel3",
    ):
        assert name in composed.source, f"{name} missing from prelude"


def test_version_directive_is_first_line(make_project, simple_image):
    _, composed = _compose(make_project({"image.glsl": simple_image}))
    assert composed.source.splitlines()[0].startswith("#version ")


def test_legacy_aliases_are_defined(make_project, simple_image):
    """Shaders copied from Shadertoy often use texture2D / iGlobalTime."""
    _, composed = _compose(make_project({"image.glsl": simple_image}))
    assert "#define texture2D texture" in composed.source
    assert "#define iGlobalTime iTime" in composed.source


def test_missing_main_image_is_reported(make_project):
    root = make_project({"image.glsl": "float x = 1.0;\n"})
    with pytest.raises(ProjectError, match="no mainImage"):
        _compose(root)


class TestLineMapping:
    def test_every_user_line_maps_back_exactly(self, make_project):
        source = "\n".join(
            f"// line {i}" for i in range(1, 21)
        ) + "\nvoid mainImage(out vec4 c, in vec2 f) { c = vec4(1.0); }\n"
        _, composed = _compose(make_project({"image.glsl": source}))

        mapped = [
            (i + 1, o)
            for i, o in enumerate(composed.origins)
            if o is not None and o.file == "image.glsl"
        ]
        assert mapped, "no lines attributed to image.glsl"
        # The offset between composed and source lines must be constant.
        offsets = {composed_line - o.line for composed_line, o in mapped}
        assert len(offsets) == 1, f"line mapping is not affine: {offsets}"

        # And the text at each mapped line must match the original.
        composed_lines = composed.source.splitlines()
        source_lines = source.split("\n")
        for composed_line, origin in mapped:
            assert composed_lines[composed_line - 1] == source_lines[origin.line - 1]

    def test_common_and_pass_are_distinguished(self, make_project, simple_image):
        _, composed = _compose(
            make_project(
                {
                    "image.glsl": simple_image,
                    "common.glsl": "float a = 1.0;\nfloat b = 2.0;\n",
                }
            )
        )
        files = {o.file for o in composed.origins if o is not None}
        assert files == {"common.glsl", "image.glsl"}

    def test_generated_lines_have_no_origin(self, make_project, simple_image):
        _, composed = _compose(make_project({"image.glsl": simple_image}))
        # The #version line is generated.
        assert composed.origin_of(1) is None

    def test_nearest_origin_falls_back_upward(self, make_project, simple_image):
        _, composed = _compose(make_project({"image.glsl": simple_image}))
        total = len(composed.origins)
        # The trailing wrapper is generated, so the direct lookup is None but
        # the nearest lookup should still find the last real user line.
        assert composed.origin_of(total) is None
        assert composed.nearest_origin(total) is not None

    def test_out_of_range_lookup_is_safe(self, make_project, simple_image):
        _, composed = _compose(make_project({"image.glsl": simple_image}))
        assert composed.origin_of(10**6) is None
        assert composed.origin_of(0) is None


class TestIncludes:
    def test_include_is_expanded_and_attributed(self, make_project, simple_image):
        root = make_project(
            {
                "image.glsl": '#include "lib/util.glsl"\n' + simple_image,
                "lib/util.glsl": "float helper() { return 2.0; }\n",
            }
        )
        _, composed = _compose(root)
        assert "helper" in composed.source
        files = {o.file for o in composed.origins if o is not None}
        assert any("util.glsl" in f for f in files)

    def test_missing_include_target(self, make_project, simple_image):
        root = make_project({"image.glsl": '#include "nope.glsl"\n' + simple_image})
        with pytest.raises(ProjectError, match="#include target not found"):
            _compose(root)

    def test_circular_include_is_detected(self, make_project, simple_image):
        root = make_project(
            {
                "image.glsl": '#include "a.glsl"\n' + simple_image,
                "a.glsl": '#include "b.glsl"\n',
                "b.glsl": '#include "a.glsl"\n',
            }
        )
        with pytest.raises(ProjectError, match="circular #include"):
            _compose(root)


def test_vertex_shader_declares_the_quad_attribute():
    assert "_st_position" in vertex_source()
    assert vertex_source(410).startswith("#version 410")
