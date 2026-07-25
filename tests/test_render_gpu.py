"""GPU integration tests: real contexts, real compilation, real pixels.

These assert the semantics that are easy to get subtly wrong -- buffer feedback
ordering, coordinate orientation, uniform plumbing -- rather than just checking
that rendering does not crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadertoy_local.inputs import InputTimeline
from shadertoy_local.project import load_project
from shadertoy_local.renderer import (
    Renderer,
    RenderSettings,
    ShaderCompileError,
)

pytestmark = pytest.mark.gpu


def _render(root, gl_context, **kwargs):
    """Compile and render one frame, returning (capture, renderer)."""
    project = load_project(root)
    settings = RenderSettings(**{"width": 64, "height": 64, **kwargs})
    renderer = Renderer(project, gl_context.ctx, settings)
    renderer.compile()
    capture = renderer.render_frame()
    return capture, renderer


class TestContext:
    def test_hardware_is_preferred_over_software(self, gl_context):
        """llvmpipe is often enumerated first; picking it would be a silent
        100x slowdown.

        Skipped when a device is pinned explicitly, so the whole suite can be
        pointed at a software rasterizer for cross-driver checks. The selection
        logic itself is covered deterministically in test_context.py.
        """
        import os

        if os.environ.get("SHADERTOY_DEVICE") is not None:
            pytest.skip("device pinned via SHADERTOY_DEVICE")
        if gl_context.device is not None:
            assert not gl_context.device.is_software

    def test_context_reports_a_usable_version(self, gl_context):
        assert gl_context.version_code >= 330


class TestCompilation:
    def test_valid_shader_compiles(self, make_project, simple_image, gl_context):
        project = load_project(make_project({"image.glsl": simple_image}))
        renderer = Renderer(project, gl_context.ctx, RenderSettings())
        try:
            assert renderer.compile() == []
        finally:
            renderer.release()

    def test_error_maps_to_the_right_source_line(
        self, make_project, gl_context
    ):
        """The end-to-end guarantee: a driver line number becomes file:line."""
        source = (
            "// a comment\n"
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
            "    fragColor = vec4(1.0);\n"
            "    fragColor.r = totally_undefined;\n"
            "}\n"
        )
        project = load_project(make_project({"image.glsl": source}))
        renderer = Renderer(project, gl_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors = [d for d in excinfo.value.diagnostics if d.is_error]
        assert errors, "expected at least one error"
        assert errors[0].file == "image.glsl"
        assert errors[0].line == 4, f"got line {errors[0].line}, expected 4"

    def test_error_in_common_attributes_to_common(self, make_project, gl_context):
        project = load_project(
            make_project(
                {
                    "common.glsl": "float bad() { return nope; }\n",
                    "image.glsl": "void mainImage(out vec4 c, in vec2 f){ c=vec4(1.0); }\n",
                }
            )
        )
        renderer = Renderer(project, gl_context.ctx, RenderSettings())
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors = [d for d in excinfo.value.diagnostics if d.is_error]
        assert errors[0].file == "common.glsl"
        assert errors[0].line == 1

    def test_collect_all_reports_every_broken_pass(self, make_project, gl_context):
        broken = "void mainImage(out vec4 c, in vec2 f){ c = missing_symbol; }\n"
        project = load_project(
            make_project({"image.glsl": broken, "buffer_a.glsl": broken})
        )
        renderer = Renderer(project, gl_context.ctx, RenderSettings())
        try:
            diagnostics = renderer.compile(collect_all=True)
        finally:
            renderer.release()
        failing = {d.pass_name for d in diagnostics if d.is_error}
        assert failing == {"image", "buffer_a"}


class TestOutputBasics:
    def test_shape_and_dtype(self, make_project, simple_image, gl_context):
        capture, renderer = _render(
            make_project({"image.glsl": simple_image}), gl_context,
            width=32, height=16,
        )
        try:
            array = capture.images["image"]
            assert array.shape == (16, 32, 4)
            assert array.dtype == np.float32
        finally:
            renderer.release()

    def test_float_targets_preserve_values_above_one(
        self, make_project, gl_context
    ):
        """An 8-bit target would clamp this to 1.0 and hide the overflow."""
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(7.5, 0.0, 0.0, 1.0); }\n"
        capture, renderer = _render(make_project({"image.glsl": source}), gl_context)
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(7.5)
        finally:
            renderer.release()

    def test_nan_reaches_the_cpu(self, make_project, gl_context):
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(sqrt(-1.0),0,0,1); }\n"
        capture, renderer = _render(make_project({"image.glsl": source}), gl_context)
        try:
            assert np.isnan(capture.images["image"][..., 0]).any()
        finally:
            renderer.release()

    def test_fragcoord_origin_is_bottom_left(self, make_project, gl_context):
        """Shadertoy's y axis points up; row 0 of our array is the bottom row."""
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(f.y < iResolution.y * 0.5 ? 1.0 : 0.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        capture, renderer = _render(make_project({"image.glsl": source}), gl_context)
        try:
            array = capture.images["image"]
            assert array[0, 0, 0] == pytest.approx(1.0), "bottom row should be red"
            assert array[-1, 0, 0] == pytest.approx(0.0), "top row should be black"
        finally:
            renderer.release()


class TestUniforms:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("iResolution.x / 100.0", 0.64),
            ("iResolution.y / 100.0", 0.32),
            ("float(iFrame) / 10.0", 0.5),
            ("iTime", 5 / 60),
            ("iFrameRate / 100.0", 0.6),
            ("iTimeDelta * 60.0", 1.0),
        ],
    )
    def test_uniform_values_reach_the_shader(
        self, make_project, gl_context, expr, expected
    ):
        source = (
            "void mainImage(out vec4 c, in vec2 f){ c = vec4(%s, 0.0, 0.0, 1.0); }\n"
            % expr
        )
        capture, renderer = _render(
            make_project({"image.glsl": source}),
            gl_context,
            width=64,
            height=32,
            frame=5,
        )
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(
                expected, abs=1e-5
            )
        finally:
            renderer.release()

    def test_idate_is_fixed_by_default(self, make_project, gl_context):
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(iDate.x/10000.0,0,0,1); }\n"
        capture, renderer = _render(make_project({"image.glsl": source}), gl_context)
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(0.2024, abs=1e-4)
        finally:
            renderer.release()

    def test_mouse_uniform(self, make_project, gl_context):
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(iMouse.x/100.0, iMouse.y/100.0, iMouse.z>0.0?1.0:0.0, 1.0);\n"
            "}\n"
        )
        capture, renderer = _render(
            make_project({"image.glsl": source}),
            gl_context,
            inputs=InputTimeline.from_spec(
                [{"frame": 0, "op": "mouse_down", "pos": [25, 50]}]
            ),
        )
        try:
            pixel = capture.images["image"][0, 0]
            assert pixel[0] == pytest.approx(0.25)
            assert pixel[1] == pytest.approx(0.50)
            assert pixel[2] == pytest.approx(1.0)
        finally:
            renderer.release()

    def test_channel_resolution_reflects_texture_size(
        self, make_project, simple_image, gl_context
    ):
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(iChannelResolution[0].x / 1000.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        root = make_project(
            {"image.glsl": source},
            config={"image": {"channels": {"0": "checker"}}},
        )
        capture, renderer = _render(root, gl_context)
        try:
            # Builtins are generated at 256x256.
            assert capture.images["image"][0, 0, 0] == pytest.approx(0.256, abs=1e-4)
        finally:
            renderer.release()


class TestKeyboardChannel:
    SOURCE = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = vec4(texelFetch(iChannel0, ivec2(32,0),0).x,\n"
        "             texelFetch(iChannel0, ivec2(32,1),0).x,\n"
        "             texelFetch(iChannel0, ivec2(87,2),0).x, 1.0);\n"
        "}\n"
    )

    def _pixel(self, make_project, gl_context, ops, frame=0):
        root = make_project(
            {"image.glsl": self.SOURCE},
            config={"image": {"channels": {"0": {"type": "keyboard"}}}},
        )
        capture, renderer = _render(
            root, gl_context, inputs=InputTimeline.from_spec(ops), frame=frame
        )
        try:
            return capture.images["image"][0, 0]
        finally:
            renderer.release()

    def test_nothing_pressed(self, make_project, gl_context):
        pixel = self._pixel(make_project, gl_context, [])
        assert list(pixel[:3]) == pytest.approx([0.0, 0.0, 0.0])

    def test_held_but_not_pressed_on_a_later_frame(self, make_project, gl_context):
        """Row 1 is 'pressed this frame', so by frame 3 the key is only held.
        This distinction was impossible to express before the timeline."""
        pixel = self._pixel(
            make_project,
            gl_context,
            [{"frame": 0, "op": "key_down", "keys": ["space"]}],
            frame=3,
        )
        assert pixel[0] == pytest.approx(1.0), "should still be held"
        assert pixel[1] == pytest.approx(0.0), "should not be pressed-this-frame"

    def test_pressed_on_the_event_frame(self, make_project, gl_context):
        pixel = self._pixel(
            make_project,
            gl_context,
            [{"frame": 2, "op": "key_down", "keys": ["space"]}],
            frame=2,
        )
        assert pixel[0] == pytest.approx(1.0)
        assert pixel[1] == pytest.approx(1.0)

    def test_released_key_is_gone(self, make_project, gl_context):
        pixel = self._pixel(
            make_project,
            gl_context,
            [
                {"frame": 0, "op": "key_down", "keys": ["space"]},
                {"frame": 2, "op": "key_up", "keys": ["space"]},
            ],
            frame=4,
        )
        assert pixel[0] == pytest.approx(0.0)

    def test_toggled_key(self, make_project, gl_context):
        pixel = self._pixel(
            make_project,
            gl_context,
            [{"frame": 0, "op": "key_toggle", "keys": ["w"]}],
        )
        assert pixel[2] == pytest.approx(1.0)


class TestBufferSemantics:
    ACCUMULATOR = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = texture(iChannel0, f/iResolution.xy) + vec4(1.0);\n"
        "}\n"
    )
    PASSTHROUGH = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = texture(iChannel0, f/iResolution.xy);\n"
        "}\n"
    )

    def _project(self, make_project):
        return make_project(
            {"buffer_a.glsl": self.ACCUMULATOR, "image.glsl": self.PASSTHROUGH},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )

    def test_buffers_start_cleared(self, make_project, gl_context):
        capture, renderer = _render(self._project(make_project), gl_context, frame=0)
        try:
            # Frame 0 reads a cleared buffer (0) and writes 1.
            assert capture.images["buffer_a"][0, 0, 0] == pytest.approx(1.0)
        finally:
            renderer.release()

    def test_self_reference_reads_previous_frame(self, make_project, gl_context):
        """The accumulator must advance by exactly one per frame."""
        project = load_project(self._project(make_project))
        renderer = Renderer(
            project, gl_context.ctx, RenderSettings(width=16, height=16, frame=5)
        )
        try:
            renderer.compile()
            values = {
                cap.frame: float(cap.images["buffer_a"][0, 0, 0])
                for cap in renderer.run(range(6))
            }
        finally:
            renderer.release()
        assert values == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}

    def test_image_sees_current_frame_of_other_buffer(
        self, make_project, gl_context
    ):
        """Cross-buffer reads see this frame's output, unlike self-reads."""
        capture, renderer = _render(self._project(make_project), gl_context, frame=3)
        try:
            buffer_value = capture.images["buffer_a"][0, 0, 0]
            image_value = capture.images["image"][0, 0, 0]
            assert buffer_value == pytest.approx(4.0)
            assert image_value == pytest.approx(buffer_value)
        finally:
            renderer.release()

    def test_buffer_scale_reduces_resolution(self, make_project, gl_context):
        root = make_project(
            {"buffer_a.glsl": self.PASSTHROUGH, "image.glsl": self.PASSTHROUGH},
            config={
                "buffer_a": {"scale": 0.5, "channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        capture, renderer = _render(root, gl_context, width=64, height=32)
        try:
            assert capture.images["buffer_a"].shape[:2] == (16, 32)
            assert capture.images["image"].shape[:2] == (32, 64)
        finally:
            renderer.release()


class TestDeterminism:
    def test_same_settings_give_identical_pixels(self, make_project, gl_context):
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    vec2 uv=f/iResolution.xy;\n"
            "    c = vec4(sin(uv.x*40.0+iTime)*0.5+0.5, uv.y, fract(iTime), 1.0);\n"
            "}\n"
        )
        root = make_project({"image.glsl": source})
        first, r1 = _render(root, gl_context, frame=17)
        try:
            second, r2 = _render(root, gl_context, frame=17)
            try:
                assert np.array_equal(first.images["image"], second.images["image"])
            finally:
                r2.release()
        finally:
            r1.release()

    def test_no_simulate_matches_simulate_for_stateless_shaders(
        self, make_project, gl_context
    ):
        """Skipping history is only safe without feedback; prove it is safe."""
        source = (
            "void mainImage(out vec4 c, in vec2 f){ c = vec4(fract(iTime),0,0,1); }\n"
        )
        root = make_project({"image.glsl": source})
        a, r1 = _render(root, gl_context, frame=9, simulate=True)
        try:
            b, r2 = _render(root, gl_context, frame=9, simulate=False)
            try:
                assert np.array_equal(a.images["image"], b.images["image"])
            finally:
                r2.release()
        finally:
            r1.release()


class TestExamples:
    """Every shipped example must compile and produce a sane frame."""

    @pytest.mark.parametrize(
        "name",
        [
            "01-plasma",
            "02-raymarch",
            "03-feedback-trail",
            "04-textured",
            "05-interactive",
            "06-portable-common",
        ],
    )
    def test_example_renders(self, name, gl_context):
        from .conftest import EXAMPLES_DIR
        from shadertoy_local.analysis import frame_stats

        root = EXAMPLES_DIR / name
        capture, renderer = _render(
            root, gl_context, width=96, height=64, frame=30,
            inputs=InputTimeline.from_spec([
                {"frame": 0, "op": "mouse_down", "pos": [48, 32]},
                {"frame": 0, "op": "key_down", "keys": ["space"]},
                {"frame": 0, "op": "key_toggle", "keys": ["g"]},
            ]),
        )
        try:
            stats = frame_stats(capture.images["image"])
        finally:
            renderer.release()
        assert stats["finite"], f"{name} produced NaN/Inf"
        assert not stats["is_uniform"], f"{name} produced a flat frame"


class TestBufferFiltering:
    """Buffer sampler settings must match shadertoy.com, since a divergence
    here changes rendered output for shaders that rely on the defaults."""

    # A 2x2 checker in the buffer; the image samples exactly between texels,
    # where linear interpolates to 0.5 and nearest snaps to a corner.
    CHECKER = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 c = floor(fragCoord / (iResolution.xy / 2.0));\n"
        "    fragColor = vec4(mod(c.x + c.y, 2.0));\n"
        "}\n"
    )
    CENTRE = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = texture(iChannel0, vec2(0.5));\n"
        "}\n"
    )

    def _sample(self, make_project, gl_context, channel):
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.CENTRE},
            config={
                "buffer_a": {"scale": 0.25, "channels": {}},
                "image": {"channels": {"0": channel}},
            },
        )
        capture, renderer = _render(root, gl_context, width=64, height=64)
        try:
            return float(capture.images["image"][1, 1, 0])
        finally:
            renderer.release()

    def test_default_is_linear(self, make_project, gl_context):
        value = self._sample(
            make_project, gl_context, {"type": "buffer", "source": "buffer_a"}
        )
        assert value == pytest.approx(0.5), "buffer default must interpolate"

    def test_explicit_nearest_snaps(self, make_project, gl_context):
        value = self._sample(
            make_project,
            gl_context,
            {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
        )
        assert value == pytest.approx(0.0)

    def test_default_matches_explicit_linear(self, make_project, gl_context):
        implicit = self._sample(
            make_project, gl_context, {"type": "buffer", "source": "buffer_a"}
        )
        explicit = self._sample(
            make_project,
            gl_context,
            {"type": "buffer", "source": "buffer_a", "filter": "linear"},
        )
        assert implicit == pytest.approx(explicit)

    # Left half black, right half white: the top mip level averages to ~0.5,
    # while level 0 at u=0.25 is 0.0. Forcing a high LOD distinguishes them.
    HALVES = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = vec4(fragCoord.x < iResolution.x * 0.5 ? 0.0 : 1.0);\n"
        "}\n"
    )
    HIGH_LOD = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    fragColor = textureLod(iChannel0, vec2(0.25, 0.5), 10.0);\n"
        "}\n"
    )

    def _high_lod(self, make_project, gl_context, filter_name):
        root = make_project(
            {"buffer_a.glsl": self.HALVES, "image.glsl": self.HIGH_LOD},
            config={
                "buffer_a": {"channels": {}},
                "image": {
                    "channels": {
                        "0": {
                            "type": "buffer",
                            "source": "buffer_a",
                            "filter": filter_name,
                        }
                    }
                },
            },
        )
        capture, renderer = _render(root, gl_context, width=64, height=64)
        try:
            return float(capture.images["image"][1, 1, 0])
        finally:
            renderer.release()

    def test_mipmap_builds_a_real_pyramid(self, make_project, gl_context):
        assert self._high_lod(make_project, gl_context, "mipmap") == pytest.approx(
            0.5, abs=0.05
        )

    def test_linear_has_no_pyramid(self, make_project, gl_context):
        """Without mipmaps a high LOD clamps to level 0, proving the previous
        test is actually measuring the pyramid rather than base-level sampling."""
        assert self._high_lod(make_project, gl_context, "linear") == pytest.approx(0.0)
