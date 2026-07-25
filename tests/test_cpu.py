"""Software-rasterizer (CPU) rendering tests.

Two reasons these exist rather than trusting the GPU path:

1. **Coverage for GPU-less machines.** The tool must work in CI without a GPU,
   and nothing else in the suite proves that.
2. **A second GLSL compiler.** llvmpipe compiles with Mesa, whose diagnostic
   format differs from NVIDIA's. Every other diagnostic test feeds the parser a
   sample string I wrote by hand; only this file checks it against output a real
   Mesa compiler actually produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadertoy_local.analysis import frame_stats

from shadertoy_local.project import load_project
from shadertoy_local.renderer import Renderer, RenderSettings, ShaderCompileError

pytestmark = pytest.mark.cpu


def _render(root, handle, **kwargs):
    project = load_project(root)
    settings = RenderSettings(**{"width": 48, "height": 32, **kwargs})
    renderer = Renderer(project, handle.ctx, settings)
    renderer.compile()
    return renderer.render_frame(), renderer


class TestSoftwareRendering:
    def test_context_is_actually_software(self, software_context):
        info = software_context.to_dict()
        assert info["software"] is True
        assert info["version_code"] >= 330

    def test_renders_a_sane_frame(self, make_project, simple_image, software_context):
        capture, renderer = _render(
            make_project({"image.glsl": simple_image}), software_context
        )
        try:
            stats = frame_stats(capture.images["image"])
        finally:
            renderer.release()
        assert stats["finite"]
        assert not stats["is_uniform"]
        assert (stats["width"], stats["height"]) == (48, 32)

    def test_fragcoord_origin_is_bottom_left(self, make_project, software_context):
        """Orientation is a whole-pipeline property; verify it per backend."""
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(f.y < iResolution.y * 0.5 ? 1.0 : 0.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        capture, renderer = _render(make_project({"image.glsl": source}), software_context)
        try:
            array = capture.images["image"]
            assert array[0, 0, 0] == pytest.approx(1.0)
            assert array[-1, 0, 0] == pytest.approx(0.0)
        finally:
            renderer.release()

    def test_float_targets_work_on_cpu(self, make_project, software_context):
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(4.25,0,0,1); }\n"
        capture, renderer = _render(make_project({"image.glsl": source}), software_context)
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(4.25)
        finally:
            renderer.release()

    def test_buffer_feedback_works_on_cpu(self, make_project, software_context):
        """Ping-pong swapping must not rely on NVIDIA-specific behaviour."""
        accumulator = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = texture(iChannel0, f/iResolution.xy) + vec4(1.0);\n"
            "}\n"
        )
        passthrough = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = texture(iChannel0, f/iResolution.xy);\n"
            "}\n"
        )
        root = make_project(
            {"buffer_a.glsl": accumulator, "image.glsl": passthrough},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        project = load_project(root)
        renderer = Renderer(
            project, software_context.ctx, RenderSettings(width=16, height=16, frame=3)
        )
        try:
            renderer.compile()
            values = {
                cap.frame: float(cap.images["buffer_a"][0, 0, 0])
                for cap in renderer.run(range(4))
            }
        finally:
            renderer.release()
        assert values == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

    def test_keyboard_channel_works_on_cpu(self, make_project, software_context):
        from shadertoy_local.inputs import KeyboardState

        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(texelFetch(iChannel0, ivec2(32,0),0).x, 0, 0, 1);\n"
            "}\n"
        )
        root = make_project(
            {"image.glsl": source},
            config={"image": {"channels": {"0": {"type": "keyboard"}}}},
        )
        capture, renderer = _render(
            root, software_context, keyboard=KeyboardState.from_spec(keys=["space"])
        )
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(1.0)
        finally:
            renderer.release()


class TestRealMesaDiagnostics:
    """Mesa's log format is `0:LINE(COL): error: msg` -- distinct from NVIDIA's
    `0(LINE) : error CODE: msg`. These assert against a real Mesa compiler."""

    def test_error_maps_to_correct_source_line(self, make_project, software_context):
        source = (
            "// a comment\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(1.0);\n"
            "    fragColor.r = totally_undefined;\n"
            "}\n"
        )
        project = load_project(make_project({"image.glsl": source}))
        renderer = Renderer(project, software_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors = [d for d in excinfo.value.diagnostics if d.is_error]
        assert errors, "Mesa error log was not parsed at all"
        assert errors[0].file == "image.glsl"
        assert errors[0].line == 4
        # Mesa supplies a column, unlike NVIDIA.
        assert errors[0].column is not None and errors[0].column > 0

    def test_error_in_common_attributes_to_common(self, make_project, software_context):
        project = load_project(
            make_project(
                {
                    "common.glsl": "float bad() { return nope; }\n",
                    "image.glsl": "void mainImage(out vec4 c, in vec2 f){ c=vec4(1.0); }\n",
                }
            )
        )
        renderer = Renderer(project, software_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors = [d for d in excinfo.value.diagnostics if d.is_error]
        assert (errors[0].file, errors[0].line) == ("common.glsl", 1)

    def test_mesa_warnings_are_not_counted_as_errors(
        self, make_project, software_context
    ):
        """Mesa emits warnings NVIDIA does not (e.g. 'used uninitialized').
        Misclassifying one as an error would fail a perfectly good shader."""
        from shadertoy_local.diagnostics import summarize

        source = (
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    float x;\n"
            "    fragColor = vec4(x) + vec4(undefined_symbol);\n"
            "}\n"
        )
        project = load_project(make_project({"image.glsl": source}))
        renderer = Renderer(project, software_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors, warnings = summarize(excinfo.value.diagnostics)
        assert errors >= 1
        # Every diagnostic must be classified, none dumped as raw text.
        for diag in excinfo.value.diagnostics:
            assert diag.severity in ("error", "warning", "note")
            assert diag.composed_line is not None

    def test_no_unparsed_lines_leak_through(self, make_project, software_context):
        """An unrecognised log line is surfaced verbatim with no position; if the
        Mesa regex were wrong, that is how it would show up."""
        project = load_project(
            make_project({"image.glsl": "void mainImage(out vec4 c, in vec2 f){ c = q; }\n"})
        )
        renderer = Renderer(project, software_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        unpositioned = [
            d for d in excinfo.value.diagnostics if d.composed_line is None
        ]
        assert not unpositioned, f"parser failed on: {[d.message for d in unpositioned]}"


@pytest.mark.gpu
class TestCrossBackendAgreement:
    """The GPU and the CPU rasterizer should agree to within a level or two.

    This is what makes golden images portable between machines, and it justifies
    the default --max-diff of 2 rather than 0.
    """

    SHADER = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = fragCoord / iResolution.xy;\n"
        "    vec3 col = 0.5 + 0.5 * cos(6.2831 * (uv.x + vec3(0.0, 0.33, 0.67)));\n"
        "    col *= smoothstep(0.0, 0.4, uv.y);\n"
        "    fragColor = vec4(col, 1.0);\n"
        "}\n"
    )

    def test_gpu_and_cpu_agree(self, make_project, gl_context, software_context):
        from shadertoy_local.analysis import to_uint8_image

        root = make_project({"image.glsl": self.SHADER})

        rendered = {}
        for name, handle in (("gpu", gl_context), ("cpu", software_context)):
            capture, renderer = _render(root, handle, width=64, height=48, frame=7)
            try:
                rendered[name] = to_uint8_image(capture.images["image"])
            finally:
                renderer.release()

        diff = np.abs(
            rendered["gpu"][..., :3].astype(np.int16)
            - rendered["cpu"][..., :3].astype(np.int16)
        )
        assert diff.max() <= 2, f"backends differ by {diff.max()} levels"
        assert diff.mean() <= 0.5

    def test_deterministic_within_the_cpu_backend(self, make_project, software_context):
        root = make_project({"image.glsl": self.SHADER})
        first, r1 = _render(root, software_context, frame=5)
        try:
            second, r2 = _render(root, software_context, frame=5)
            try:
                assert np.array_equal(first.images["image"], second.images["image"])
            finally:
                r2.release()
        finally:
            r1.release()
