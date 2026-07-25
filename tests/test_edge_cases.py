"""Edge cases and deliberately broken inputs.

Written to probe for real failures rather than to raise a count: every case here
is something a user or agent could plausibly do that the happy path never
exercises -- empty files, corrupt assets, degenerate sizes, nonsensical event
orders, and hostile paths.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from shadertoy_local.compose import compose_pass
from shadertoy_local.inputs import InputError, InputTimeline
from shadertoy_local.project import ProjectError, find_project_root, load_project

IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"
PASSTHROUGH = (
    "void mainImage(out vec4 c, in vec2 f){\n"
    "    c = texture(iChannel0, f/iResolution.xy);\n"
    "}\n"
)


# --------------------------------------------------------------------------
# Source files
# --------------------------------------------------------------------------


class TestDegenerateSourceFiles:
    def test_empty_image_file(self, make_project):
        root = make_project({"image.glsl": ""})
        project = load_project(root)
        with pytest.raises(ProjectError, match="no mainImage"):
            compose_pass(project, project.passes["image"])

    def test_whitespace_only_image_file(self, make_project):
        root = make_project({"image.glsl": "\n\n   \n\t\n"})
        project = load_project(root)
        with pytest.raises(ProjectError, match="no mainImage"):
            compose_pass(project, project.passes["image"])

    def test_comment_only_image_file(self, make_project):
        root = make_project({"image.glsl": "// nothing here\n/* nor here */\n"})
        project = load_project(root)
        with pytest.raises(ProjectError, match="no mainImage"):
            compose_pass(project, project.passes["image"])

    def test_main_image_inside_a_comment_does_not_count(self, make_project):
        """The detector must not be fooled by a commented-out definition."""
        root = make_project(
            {"image.glsl": "// void mainImage(out vec4 c, in vec2 f) {}\n"}
        )
        project = load_project(root)
        with pytest.raises(ProjectError, match="no mainImage"):
            compose_pass(project, project.passes["image"])

    def test_non_utf8_source_is_reported_clearly(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "image.glsl").write_bytes(b"void mainImage(){} // \xff\xfe invalid")
        with pytest.raises(ProjectError, match="not valid UTF-8"):
            load_project(root)

    def test_source_with_bom_still_compiles_to_a_pass(self, make_project):
        """A BOM is common from Windows editors; it must not break detection."""
        root = make_project({"image.glsl": "\ufeff" + IMAGE})
        project = load_project(root)
        composed = compose_pass(project, project.passes["image"])
        assert "mainImage" in composed.source

    def test_crlf_line_endings_keep_the_line_map_correct(self, make_project):
        source = "// one\r\n// two\r\nvoid mainImage(out vec4 c, in vec2 f){ c=vec4(1.0); }\r\n"
        root = make_project({"image.glsl": source})
        project = load_project(root)
        composed = compose_pass(project, project.passes["image"])
        origins = [o for o in composed.origins if o and o.file == "image.glsl"]
        assert [o.line for o in origins[:3]] == [1, 2, 3]


class TestPassDiscoveryEdges:
    def test_buffer_d_without_a_b_c(self, make_project):
        """Buffers need not be contiguous."""
        root = make_project({"image.glsl": IMAGE, "buffer_d.glsl": IMAGE})
        project = load_project(root)
        assert [p.name for p in project.ordered_passes] == ["buffer_d", "image"]

    def test_frag_extension_is_accepted(self, make_project):
        root = make_project({"image.frag": IMAGE})
        assert "image" in load_project(root).passes

    def test_glsl_wins_over_frag(self, make_project):
        root = make_project({"image.glsl": IMAGE, "image.frag": "broken"})
        assert load_project(root).passes["image"].path.name == "image.glsl"

    def test_configured_pass_without_a_file(self, make_project):
        root = make_project({"image.glsl": IMAGE}, config={"buffer_a": {"scale": 0.5}})
        with pytest.raises(ProjectError, match="no such pass file"):
            load_project(root)

    def test_find_root_accepts_a_file_path(self, make_project):
        root = make_project({"image.glsl": IMAGE})
        assert find_project_root(root / "image.glsl") == root

    def test_config_without_image_pass(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "shadertoy.json").write_text("{}")
        with pytest.raises(ProjectError, match="no image pass"):
            load_project(root, search_parents=False)

    def test_unicode_project_path(self, tmp_path):
        root = tmp_path / "shädér tøy ✨"
        root.mkdir()
        (root / "image.glsl").write_text(IMAGE, encoding="utf-8")
        assert load_project(root).passes["image"].source == IMAGE


class TestIncludeEdges:
    def test_include_from_a_subdirectory(self, make_project):
        root = make_project(
            {
                "image.glsl": '#include "lib/a.glsl"\n' + IMAGE,
                "lib/a.glsl": '#include "lib/b.glsl"\n',
                "lib/b.glsl": "float helper(){ return 1.0; }\n",
            }
        )
        project = load_project(root)
        composed = compose_pass(project, project.passes["image"])
        assert "helper" in composed.source

    def test_include_depth_limit(self, make_project):
        """A long chain must hit the limit rather than recursing forever."""
        files = {"image.glsl": '#include "d0.glsl"\n' + IMAGE}
        for i in range(30):
            files[f"d{i}.glsl"] = f'#include "d{i + 1}.glsl"\n'
        files["d30.glsl"] = "float x = 1.0;\n"
        project = load_project(make_project(files))
        with pytest.raises(ProjectError, match="nested deeper than"):
            compose_pass(project, project.passes["image"])

    def test_self_include_is_circular(self, make_project):
        root = make_project({"image.glsl": '#include "image.glsl"\n' + IMAGE})
        project = load_project(root)
        with pytest.raises(ProjectError, match="circular #include"):
            compose_pass(project, project.passes["image"])

    def test_include_of_non_utf8_file(self, make_project, tmp_path):
        root = make_project({"image.glsl": '#include "bad.glsl"\n' + IMAGE})
        (root / "bad.glsl").write_bytes(b"\xff\xfe\x00")
        project = load_project(root)
        with pytest.raises(ProjectError, match="not valid UTF-8"):
            compose_pass(project, project.passes["image"])

    def test_include_the_same_file_twice_is_not_circular(self, make_project):
        """Diamond includes are legal; only cycles are not."""
        root = make_project(
            {
                "image.glsl": '#include "a.glsl"\n#include "b.glsl"\n' + IMAGE,
                "a.glsl": '#include "shared.glsl"\n',
                "b.glsl": '#include "shared.glsl"\n',
                "shared.glsl": "// shared\n",
            }
        )
        project = load_project(root)
        composed = compose_pass(project, project.passes["image"])
        assert composed.source.count("// shared") == 2

    def test_indented_include_is_honoured(self, make_project):
        root = make_project(
            {"image.glsl": '  #  include "a.glsl"\n' + IMAGE, "a.glsl": "// inc\n"}
        )
        project = load_project(root)
        composed = compose_pass(project, project.passes["image"])
        assert "// inc" in composed.source


# --------------------------------------------------------------------------
# Input timeline
# --------------------------------------------------------------------------


class TestTimelineEdges:
    def test_key_up_without_a_prior_down(self):
        """Nonsense but harmless: releasing an unheld key is a no-op."""
        line = InputTimeline.from_spec(
            [{"frame": 0, "op": "key_up", "keys": ["space"]}]
        )
        assert line.state_at(0).held == frozenset()

    def test_mouse_up_without_a_prior_down(self):
        line = InputTimeline.from_spec([{"frame": 0, "op": "mouse_up"}])
        state = line.state_at(0)
        assert state.button_down is False
        assert state.press_frame is None

    def test_double_press_without_release(self):
        line = InputTimeline.from_spec(
            [
                {"frame": 0, "op": "mouse_down", "pos": [10, 10]},
                {"frame": 5, "op": "mouse_down", "pos": [50, 50]},
            ]
        )
        state = line.state_at(5)
        assert state.press_frame == 5
        assert (state.click_x, state.click_y) == (50.0, 50.0)

    def test_duplicate_identical_events_are_idempotent(self):
        ops = [{"frame": 0, "op": "key_down", "keys": ["a"]}] * 3
        assert InputTimeline.from_spec(ops).state_at(0).held == frozenset({65})

    def test_key_down_and_key_up_same_frame_file_order_wins(self):
        down_then_up = InputTimeline.from_spec(
            [
                {"frame": 0, "op": "key_down", "keys": ["a"]},
                {"frame": 0, "op": "key_up", "keys": ["a"]},
            ]
        )
        up_then_down = InputTimeline.from_spec(
            [
                {"frame": 0, "op": "key_up", "keys": ["a"]},
                {"frame": 0, "op": "key_down", "keys": ["a"]},
            ]
        )
        assert down_then_up.state_at(0).held == frozenset()
        assert up_then_down.state_at(0).held == frozenset({65})

    def test_pressed_row_still_fires_when_released_the_same_frame(self):
        """The press happened, so row 1 reports it even though row 0 does not."""
        line = InputTimeline.from_spec(
            [
                {"frame": 0, "op": "key_down", "keys": ["a"]},
                {"frame": 0, "op": "key_up", "keys": ["a"]},
            ]
        )
        state = line.state_at(0)
        assert 65 in state.pressed
        assert 65 not in state.held

    def test_repeated_toggles_flip_each_time(self):
        ops = [{"frame": i, "op": "key_toggle", "keys": ["g"]} for i in range(5)]
        line = InputTimeline.from_spec(ops)
        assert [71 in line.state_at(f).toggled for f in range(5)] == [
            True, False, True, False, True
        ]

    def test_untoggle_an_untoggled_key_is_a_no_op(self):
        line = InputTimeline.from_spec(
            [{"frame": 0, "op": "key_untoggle", "keys": ["g"]}]
        )
        assert line.state_at(0).toggled == frozenset()

    def test_state_before_any_event(self):
        line = InputTimeline.from_spec([{"frame": 100, "op": "mouse_down", "pos": [1, 1]}])
        state = line.state_at(0)
        assert not state.button_down and (state.x, state.y) == (0.0, 0.0)

    def test_state_far_after_the_last_event(self):
        line = InputTimeline.from_spec([{"frame": 1, "op": "key_down", "keys": ["a"]}])
        assert 65 in line.state_at(10_000).held

    def test_very_large_frame_index(self):
        line = InputTimeline.from_spec(
            [{"frame": 10**9, "op": "key_down", "keys": ["a"]}]
        )
        assert line.last_frame == 10**9
        assert line.state_at(10**9 - 1).held == frozenset()

    def test_fractional_fps_time_conversion(self):
        line = InputTimeline.from_spec(
            [{"time": 1.0, "op": "mouse_up"}], fps=29.97
        )
        assert line.events[0].frame == 30

    def test_many_events_on_one_frame(self):
        ops = [
            {"frame": 0, "op": "key_down", "keys": [code]} for code in range(65, 91)
        ]
        state = InputTimeline.from_spec(ops).state_at(0)
        assert len(state.held) == 26

    def test_empty_list_is_valid(self):
        line = InputTimeline.from_spec([])
        assert not line.active
        assert line.state_at(0).keyboard_bytes() == bytes(256 * 3)

    def test_duplicate_keys_within_one_op_are_deduplicated(self):
        line = InputTimeline.from_spec(
            [{"frame": 0, "op": "key_down", "keys": ["a", "a", "A"]}]
        )
        assert line.events[0].keys == (65,)

    def test_normalized_positions_outside_zero_one(self):
        """Off-screen coordinates are legal; shaders may rely on them."""
        line = InputTimeline.from_spec(
            [{"frame": 0, "op": "mouse_move", "pos": [1.5, -0.25], "normalized": True}]
        )
        x, y, _, _ = line.state_at(0).mouse_vec4(640, 360, 0)
        assert (x, y) == (960.0, -90.0)

    def test_pos_as_string_is_accepted(self):
        line = InputTimeline.from_spec(
            [{"frame": 0, "op": "mouse_move", "pos": "10, 20"}]
        )
        assert (line.state_at(0).x, line.state_at(0).y) == (10.0, 20.0)

    @pytest.mark.parametrize(
        "bad",
        [
            [{"frame": 0, "op": "mouse_move", "pos": ["a", "b"]}],
            [{"frame": 0, "op": "mouse_move", "pos": {}}],
            [{"frame": 0, "op": "key_down", "keys": [None]}],
            [{"frame": 0, "op": "key_down", "keys": [1.5]}],
            [{"frame": 0, "op": None}],
            [None],
            [[]],
        ],
    )
    def test_malformed_operations(self, bad):
        with pytest.raises(InputError):
            InputTimeline.from_spec(bad)

    def test_json_object_at_top_level_is_one_event(self):
        line = InputTimeline.from_json('{"frame": 3, "op": "key_down", "keys": ["a"]}')
        assert 65 in line.state_at(3).held

    def test_json_number_at_top_level_is_rejected(self):
        with pytest.raises(InputError, match="must be an array"):
            InputTimeline.from_json("42")


# --------------------------------------------------------------------------
# Assets and analysis
# --------------------------------------------------------------------------


class TestTextureAssetEdges:
    def _project_with_texture(self, make_project, name="tex.png"):
        return make_project(
            {"image.glsl": PASSTHROUGH},
            config={"image": {"channels": {"0": {"type": "texture", "source": name}}}},
        )

    def test_corrupt_image_file(self, make_project):
        from shadertoy_local.channels import load_image_array

        root = self._project_with_texture(make_project)
        (root / "tex.png").write_bytes(b"not a png at all")
        load_project(root)  # config resolves: the file exists
        with pytest.raises(ProjectError, match="could not read texture"):
            load_image_array(root / "tex.png")

    def test_zero_byte_image_file(self, make_project):
        from shadertoy_local.channels import load_image_array

        root = self._project_with_texture(make_project)
        (root / "tex.png").write_bytes(b"")
        with pytest.raises(ProjectError, match="could not read texture"):
            load_image_array(root / "tex.png")

    @pytest.mark.parametrize("mode", ["L", "LA", "P", "RGB", "RGBA", "1"])
    def test_unusual_image_modes_are_converted(self, tmp_path, mode):
        """Anything Pillow can open must arrive as RGBA."""
        from PIL import Image

        from shadertoy_local.channels import load_image_array

        path = tmp_path / f"{mode}.png"
        Image.new(mode, (4, 3)).save(path)
        array = load_image_array(path)
        assert array.shape == (3, 4, 4)
        assert array.dtype == np.uint8

    def test_directory_given_as_a_texture(self, make_project):
        root = make_project(
            {"image.glsl": PASSTHROUGH},
            config={"image": {"channels": {"0": {"type": "texture", "source": "sub"}}}},
        )
        (root / "sub").mkdir()
        with pytest.raises(ProjectError, match="not found"):
            load_project(root)

    def test_texture_escaping_the_project_root_still_resolves(self, make_project):
        """Not forbidden, but it must resolve predictably rather than silently
        producing a path inside the project."""
        from PIL import Image

        root = make_project({"image.glsl": PASSTHROUGH})
        outside = root.parent / "outside.png"
        Image.new("RGBA", (2, 2)).save(outside)
        (root / "shadertoy.json").write_text(
            json.dumps(
                {"image": {"channels": {"0": {"type": "texture", "source": "../outside.png"}}}}
            )
        )
        binding = load_project(root).passes["image"].channels[0]
        assert binding.path == outside.resolve()


class TestBuiltinEdges:
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 17])
    def test_tiny_and_odd_builtin_sizes(self, size):
        from shadertoy_local.channels import builtin_array

        for name in ("noise", "checker", "uv", "gradient", "bayer", "blue-noise"):
            array = builtin_array(name, size)
            assert array.shape == (size, size, 4), (name, size)
            assert array.dtype == np.uint8

    def test_unknown_builtin_lists_the_options(self):
        from shadertoy_local.channels import builtin_array

        with pytest.raises(ProjectError, match="unknown builtin texture"):
            builtin_array("perlin")

    def test_builtins_are_reproducible(self):
        """Golden tests depend on this."""
        from shadertoy_local.channels import builtin_array

        for name in ("noise", "gray-noise", "blue-noise", "bayer"):
            assert np.array_equal(builtin_array(name, 32), builtin_array(name, 32))

    def test_bayer_is_a_permutation(self):
        """Every threshold level appears exactly once in an ordered-dither tile."""
        from shadertoy_local.channels import builtin_array

        values = builtin_array("bayer", 4)[..., 0].ravel()
        assert len(set(values.tolist())) == 16


class TestAnalysisEdges:
    def test_one_pixel_frame(self):
        from shadertoy_local.analysis import frame_stats, parse_probe, run_probe

        array = np.array([[[0.25, 0.5, 0.75, 1.0]]], dtype=np.float32)
        stats = frame_stats(array)
        assert stats["pixels"] == 1
        assert stats["is_uniform"] is True
        result = run_probe(array, parse_probe("0,0"))
        assert result["rgba"] == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_probe_on_negative_coordinates_clamps(self):
        from shadertoy_local.analysis import Probe, run_probe

        array = np.zeros((4, 4, 4), dtype=np.float32)
        array[0, 0] = [1.0, 0.0, 0.0, 1.0]
        assert run_probe(array, Probe(x=-5, y=-5))["rgba"][0] == pytest.approx(1.0)

    def test_extreme_float_values(self):
        from shadertoy_local.analysis import frame_stats, to_uint8_image

        array = np.array(
            [[[1e30, -1e30, 0.0, 1.0]]], dtype=np.float32
        )
        stats = frame_stats(array)
        assert stats["finite"] is True
        assert stats["fraction_above_one"] > 0
        assert list(to_uint8_image(array)[0, 0]) == [255, 0, 0, 255]

    def test_histogram_with_one_bin(self):
        from shadertoy_local.analysis import histogram

        array = np.zeros((4, 4, 4), dtype=np.float32)
        assert histogram(array, bins=1)["r"] == [16]

    def test_mixed_nan_and_inf_counts(self):
        from shadertoy_local.analysis import frame_stats

        array = np.zeros((2, 2, 4), dtype=np.float32)
        array[0, 0, 0] = np.nan
        array[0, 1, 1] = np.inf
        array[1, 0, 2] = -np.inf
        stats = frame_stats(array)
        assert stats["nan_count"] == 1
        assert stats["inf_count"] == 2
        assert stats["finite"] is False

    def test_probe_expectation_with_more_components_than_given(self):
        from shadertoy_local.analysis import parse_probe, run_probe

        array = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        # Only the red channel is asserted.
        assert run_probe(array, parse_probe("0,0=1"))["passed"] is True


class TestGoldenEdges:
    def test_corrupt_golden_file(self, tmp_path):
        from shadertoy_local.golden import compare, golden_path

        path = golden_path(tmp_path, "image_f0000")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"definitely not a png")
        with pytest.raises(Exception):
            compare(np.zeros((4, 4, 4), dtype=np.float32), tmp_path, "image_f0000")

    def test_golden_with_nan_in_the_render(self, tmp_path):
        """Non-finite pixels become 0 in the 8-bit view, so comparison still
        works; frame_stats is what detects them."""
        from shadertoy_local.golden import compare, write_golden

        clean = np.zeros((4, 4, 4), dtype=np.float32)
        write_golden(clean, tmp_path, "k")
        broken = clean.copy()
        broken[0, 0, 0] = np.nan
        assert compare(broken, tmp_path, "k").passed

    def test_one_pixel_golden(self, tmp_path):
        from shadertoy_local.golden import compare, write_golden

        array = np.full((1, 1, 4), 0.5, dtype=np.float32)
        write_golden(array, tmp_path, "k")
        assert compare(array, tmp_path, "k").passed


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@pytest.mark.gpu
class TestRenderEdges:
    def _render(self, root, gl_context, **kwargs):
        from shadertoy_local.renderer import Renderer, RenderSettings

        settings = RenderSettings(**{"width": 8, "height": 8, **kwargs})
        renderer = Renderer(load_project(root), gl_context.ctx, settings)
        renderer.compile()
        return renderer.render_frame(), renderer

    def test_one_by_one_render(self, make_project, gl_context):
        capture, renderer = self._render(
            make_project({"image.glsl": IMAGE}), gl_context, width=1, height=1
        )
        try:
            assert capture.images["image"].shape == (1, 1, 4)
        finally:
            renderer.release()

    def test_extremely_wide_aspect(self, make_project, gl_context):
        capture, renderer = self._render(
            make_project({"image.glsl": IMAGE}), gl_context, width=1024, height=1
        )
        try:
            assert capture.images["image"].shape == (1, 1024, 4)
        finally:
            renderer.release()

    def test_tiny_buffer_scale_clamps_to_one_pixel(self, make_project, gl_context):
        """floor(8 * 0.01) is 0, which would be an invalid texture size."""
        root = make_project(
            {"image.glsl": PASSTHROUGH, "buffer_a.glsl": IMAGE},
            config={
                "buffer_a": {"scale": 0.01},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        capture, renderer = self._render(root, gl_context, width=8, height=8)
        try:
            assert capture.images["buffer_a"].shape[:2] == (1, 1)
        finally:
            renderer.release()

    def test_scale_of_exactly_one(self, make_project, gl_context):
        root = make_project(
            {"image.glsl": PASSTHROUGH, "buffer_a.glsl": IMAGE},
            config={
                "buffer_a": {"scale": 1.0},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        capture, renderer = self._render(root, gl_context, width=16, height=8)
        try:
            assert capture.images["buffer_a"].shape[:2] == (8, 16)
        finally:
            renderer.release()

    def test_all_four_buffers_chained(self, make_project, gl_context):
        """A -> B -> C -> D -> Image, each adding one, exercises pass ordering."""
        add = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = texture(iChannel0, f/iResolution.xy) + vec4(1.0);\n"
            "}\n"
        )
        root = make_project(
            {
                "buffer_a.glsl": "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n",
                "buffer_b.glsl": add,
                "buffer_c.glsl": add,
                "buffer_d.glsl": add,
                "image.glsl": PASSTHROUGH,
            },
            config={
                "buffer_b": {"channels": {"0": "buffer_a"}},
                "buffer_c": {"channels": {"0": "buffer_b"}},
                "buffer_d": {"channels": {"0": "buffer_c"}},
                "image": {"channels": {"0": "buffer_d"}},
            },
        )
        capture, renderer = self._render(root, gl_context, precharge=0)
        try:
            # Each pass sees the current frame's upstream output: 1, 2, 3, 4.
            assert capture.images["image"][0, 0, 0] == pytest.approx(4.0)
        finally:
            renderer.release()

    def test_unbound_channel_reads_as_zero(self, make_project, gl_context):
        """Sampling an unbound iChannel must be defined, not garbage."""
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = texture(iChannel3, vec2(0.5)) + vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        capture, renderer = self._render(
            make_project({"image.glsl": source}), gl_context
        )
        try:
            assert np.isfinite(capture.images["image"]).all()
        finally:
            renderer.release()

    def test_channel_resolution_is_zero_for_unbound(self, make_project, gl_context):
        source = (
            "void mainImage(out vec4 c, in vec2 f){\n"
            "    c = vec4(iChannelResolution[3].x, 0, 0, 1);\n"
            "}\n"
        )
        capture, renderer = self._render(
            make_project({"image.glsl": source}), gl_context
        )
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(0.0)
        finally:
            renderer.release()

    def test_max_frames_boundary_is_inclusive(self, make_project, gl_context):
        """Rendering exactly the limit is allowed; one more is not."""
        from shadertoy_local.renderer import RenderError

        root = make_project(
            {"image.glsl": PASSTHROUGH, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        capture, renderer = self._render(root, gl_context, frame=9, max_frames=10)
        renderer.release()
        assert capture.frame == 9

        with pytest.raises(RenderError, match="would render 11 frames"):
            self._render(root, gl_context, frame=10, max_frames=10)

    def test_negative_capture_frame_is_rejected(self, make_project, gl_context):
        from shadertoy_local.renderer import Renderer, RenderError, RenderSettings

        renderer = Renderer(
            load_project(make_project({"image.glsl": IMAGE})),
            gl_context.ctx,
            RenderSettings(width=4, height=4),
        )
        try:
            renderer.compile()
            with pytest.raises(RenderError, match="must be >= 0"):
                list(renderer.run([-1]))
        finally:
            renderer.release()

    def test_render_before_compile_is_an_error(self, make_project, gl_context):
        from shadertoy_local.renderer import Renderer, RenderError, RenderSettings

        renderer = Renderer(
            load_project(make_project({"image.glsl": IMAGE})),
            gl_context.ctx,
            RenderSettings(),
        )
        try:
            with pytest.raises(RenderError, match="no passes compiled"):
                renderer.render_frame()
        finally:
            renderer.release()

    def test_buffer_reading_a_pass_that_failed_to_compile(
        self, make_project, gl_context
    ):
        """With collect_all, a good pass must not silently sample a dead one."""
        from shadertoy_local.renderer import Renderer, RenderError, RenderSettings

        root = make_project(
            {
                "buffer_a.glsl": "void mainImage(out vec4 c, in vec2 f){ c = nope; }\n",
                "image.glsl": PASSTHROUGH,
            },
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        renderer = Renderer(
            load_project(root), gl_context.ctx, RenderSettings(width=4, height=4)
        )
        try:
            renderer.compile(collect_all=True)
            with pytest.raises(RenderError, match="failed to compile"):
                list(renderer.run([0]))
        finally:
            renderer.release()

    def test_time_override_applies_only_to_the_captured_frame(
        self, make_project, gl_context
    ):
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(iTime,0,0,1); }\n"
        capture, renderer = self._render(
            make_project({"image.glsl": source}), gl_context, frame=10, time=42.0
        )
        try:
            assert capture.images["image"][0, 0, 0] == pytest.approx(42.0)
        finally:
            renderer.release()

    def test_release_is_idempotent(self, make_project, gl_context):
        from shadertoy_local.renderer import Renderer, RenderSettings

        renderer = Renderer(
            load_project(make_project({"image.glsl": IMAGE})),
            gl_context.ctx,
            RenderSettings(width=4, height=4),
        )
        renderer.compile()
        renderer.render_frame()
        renderer.release()
        renderer.release()  # must not raise


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run(capsys, *argv):
    from shadertoy_local.cli import main

    code = main(list(argv))
    captured = capsys.readouterr()
    payload = None
    if captured.out.strip():
        try:
            payload = json.loads(captured.out)
        except json.JSONDecodeError:
            payload = None
    return code, payload, captured.err


class TestResolutionParsing:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("640x360", (640, 360)),
            ("640X360", (640, 360)),
            ("640,360", (640, 360)),
            ("1x1", (1, 1)),
        ],
    )
    def test_accepted_forms(self, spec, expected):
        from shadertoy_local.cli import _parse_resolution

        assert _parse_resolution(spec) == expected

    @pytest.mark.parametrize(
        "spec", ["640", "640x", "x360", "0x100", "100x0", "-5x10", "axb", "", "640x360x1"]
    )
    def test_rejected_forms(self, spec):
        from shadertoy_local.cli import _parse_resolution

        with pytest.raises(ValueError):
            _parse_resolution(spec)


@pytest.mark.gpu
class TestCliEdges:
    def test_output_path_is_an_existing_file(
        self, capsys, make_project, tmp_path
    ):
        """Writing into a path that is a file must fail clearly, not traceback."""
        root = make_project({"image.glsl": IMAGE})
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file")
        code, _, err = _run(
            capsys, "render", "-C", str(root), "-r", "8x8", "-o", str(blocker), "--json"
        )
        assert code != 0
        assert "error" in err.lower() or "Error" in err

    def test_unicode_project_directory(self, capsys, tmp_path):
        root = tmp_path / "проект ✨"
        root.mkdir()
        (root / "image.glsl").write_text(IMAGE, encoding="utf-8")
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "-r", "8x8", "--no-write", "--json"
        )
        assert code == 0
        assert payload["ok"] is True

    def test_time_and_count_together(self, capsys, make_project):
        """--time pins only the first captured frame; the rest advance normally."""
        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(iTime,0,0,1); }\n"
        root = make_project({"image.glsl": source})
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "-r", "4x4", "--no-write",
            "--count", "2", "--every", "6", "--time", "9.0", "--stats", "--json",
        )
        assert code == 0
        maxima = [
            f["passes"]["image"]["stats"]["channels"]["r"]["max"]
            for f in payload["frames"]
        ]
        assert maxima[0] == pytest.approx(9.0)
        assert maxima[1] == pytest.approx(6 / 60)

    def test_invalid_json_on_stdin(self, capsys, make_project, monkeypatch):
        import io

        root = make_project({"image.glsl": IMAGE})
        monkeypatch.setattr("sys.stdin", io.StringIO("[{oops}]"))
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "--input", "-", "--no-write", "--json"
        )
        assert code == 2
        assert "invalid input JSON" in payload["error"]

    def test_input_file_that_does_not_exist(self, capsys, make_project):
        root = make_project({"image.glsl": IMAGE})
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "--input", "nope.json",
            "--no-write", "--json",
        )
        assert code == 2
        assert "neither inline JSON" in payload["error"]

    def test_probe_a_buffer_pass(self, capsys, make_project):
        root = make_project(
            {"image.glsl": PASSTHROUGH, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        code, payload, _ = _run(
            capsys, "probe", "-C", str(root), "-r", "8x8",
            "-p", "buffer_a", "--at", "1,1", "--json",
        )
        assert code == 0
        assert "buffer_a" in payload["passes"]

    def test_keep_alpha_preserves_transparency(
        self, capsys, make_project, tmp_path
    ):
        from shadertoy_local.analysis import load_png

        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0,0.0,0.0,0.25); }\n"
        root = make_project({"image.glsl": source})
        out = tmp_path / "a"
        _run(
            capsys, "render", "-C", str(root), "-r", "4x4", "-o", str(out),
            "--keep-alpha", "--json",
        )
        assert load_png(out / "image.png")[0, 0, 3] == 64

        out2 = tmp_path / "b"
        _run(capsys, "render", "-C", str(root), "-r", "4x4", "-o", str(out2), "--json")
        assert load_png(out2 / "image.png")[0, 0, 3] == 255

    def test_quiet_suppresses_prose_but_not_errors(self, capsys, make_project):
        broken = "void mainImage(out vec4 c, in vec2 f){ c = missing; }\n"
        root = make_project({"image.glsl": broken})
        code, _, err = _run(capsys, "check", "-C", str(root), "-q")
        assert code == 1
        assert "image.glsl" in err, "an error must survive --quiet"

    def test_zero_resolution_is_rejected(self, capsys, make_project):
        root = make_project({"image.glsl": IMAGE})
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "-w", "0", "--no-write", "--json"
        )
        assert code == 2
        assert "positive" in payload["error"]

    def test_bad_date_spec(self, capsys, make_project):
        root = make_project({"image.glsl": IMAGE})
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "--date", "2024,1", "--no-write", "--json"
        )
        assert code == 2
        assert "four numbers" in payload["error"]

    def test_render_from_a_subdirectory(self, capsys, make_project):
        """Project discovery walks upward, like git."""
        root = make_project({"image.glsl": IMAGE})
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        code, payload, _ = _run(
            capsys, "render", "-C", str(nested), "-r", "4x4", "--no-write", "--json"
        )
        assert code == 0
        assert payload["ok"] is True


# --------------------------------------------------------------------------
# Sampler independence and shared-buffer state
# --------------------------------------------------------------------------


class TestBufferSamplerSettingsAreSharedPerBuffer:
    """On shadertoy.com a buffer's filter and wrap belong to the *buffer*, not to
    the channel reading it: changing them on one reference changes every
    reference, because GL stores sampler state on the texture object.

    A config asking for two different settings for one buffer is therefore
    inexpressible on the real site. It is rejected rather than resolved in favour
    of whichever binding happens to be applied last.
    """

    CHECKER = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    vec2 g = floor(f / (iResolution.xy / 2.0));\n"
        "    c = vec4(mod(g.x + g.y, 2.0));\n"
        "}\n"
    )
    TWO_CHANNELS = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = vec4(texture(iChannel0, vec2(0.5)).r,\n"
        "             texture(iChannel1, vec2(0.5)).r, 0.0, 1.0);\n"
        "}\n"
    )

    def test_conflicting_filters_on_one_pass_are_rejected(self, make_project):
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
                        "1": {"type": "buffer", "source": "buffer_a", "filter": "linear"},
                    }
                }
            },
        )
        with pytest.raises(ProjectError, match="belong to the buffer"):
            load_project(root)

    def test_conflicting_filters_across_passes_are_rejected(self, make_project):
        root = make_project(
            {
                "buffer_a.glsl": self.CHECKER,
                "buffer_b.glsl": (
                    "void mainImage(out vec4 c, in vec2 f){\n"
                    "    c = vec4(texture(iChannel0, vec2(0.5)).r);\n"
                    "}\n"
                ),
                "image.glsl": self.TWO_CHANNELS,
            },
            config={
                "buffer_b": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "nearest"}
                    }
                },
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "linear"},
                        "1": {"type": "buffer", "source": "buffer_b"},
                    }
                },
            },
        )
        with pytest.raises(ProjectError, match="every reference"):
            load_project(root)

    def test_conflicting_wrap_is_rejected(self, make_project):
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "wrap": "clamp"},
                        "1": {"type": "buffer", "source": "buffer_a", "wrap": "repeat"},
                    }
                }
            },
        )
        with pytest.raises(ProjectError, match="belong to the buffer"):
            load_project(root)

    def test_the_error_names_both_offending_channels(self, make_project):
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
                        "1": {"type": "buffer", "source": "buffer_a", "filter": "mipmap"},
                    }
                }
            },
        )
        with pytest.raises(ProjectError) as excinfo:
            load_project(root)
        message = str(excinfo.value)
        assert "channel0" in message and "channel1" in message
        assert "nearest" in message and "mipmap" in message

    def test_matching_settings_everywhere_are_accepted(self, make_project):
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
                        "1": {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
                    }
                }
            },
        )
        project = load_project(root)
        assert all(
            b.filter == "nearest" for b in project.passes["image"].channels.values()
        )

    def test_defaults_do_not_collide_with_an_explicit_match(self, make_project):
        """One reference stating the default explicitly must not look like a
        conflict against another that omits it."""
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "image": {
                    "channels": {
                        "0": "buffer_a",
                        "1": {"type": "buffer", "source": "buffer_a",
                              "filter": "linear", "wrap": "clamp"},
                    }
                }
            },
        )
        assert load_project(root) is not None

    def test_different_buffers_may_differ(self, make_project):
        """The constraint is per buffer, not global."""
        root = make_project(
            {
                "buffer_a.glsl": self.CHECKER,
                "buffer_b.glsl": self.CHECKER,
                "image.glsl": self.TWO_CHANNELS,
            },
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "filter": "nearest"},
                        "1": {"type": "buffer", "source": "buffer_b", "filter": "linear"},
                    }
                }
            },
        )
        channels = load_project(root).passes["image"].channels
        assert (channels[0].filter, channels[1].filter) == ("nearest", "linear")

    def test_vflip_on_a_buffer_is_rejected(self, make_project):
        """It was silently ignored: buffers are never flipped by this renderer."""
        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": PASSTHROUGH},
            config={
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a", "vflip": True}
                    }
                }
            },
        )
        with pytest.raises(ProjectError, match='"vflip" is not supported for buffer'):
            load_project(root)


@pytest.mark.gpu
class TestSharedBufferSamplerApplies:
    """With settings validated as consistent, they must actually take effect on
    every reference."""

    CHECKER = TestBufferSamplerSettingsAreSharedPerBuffer.CHECKER
    TWO_CHANNELS = TestBufferSamplerSettingsAreSharedPerBuffer.TWO_CHANNELS

    def _pixel(self, make_project, gl_context, filter_name):
        from shadertoy_local.renderer import Renderer, RenderSettings

        root = make_project(
            {"buffer_a.glsl": self.CHECKER, "image.glsl": self.TWO_CHANNELS},
            config={
                "buffer_a": {"scale": 0.25},
                "image": {
                    "channels": {
                        "0": {"type": "buffer", "source": "buffer_a",
                              "filter": filter_name},
                        "1": {"type": "buffer", "source": "buffer_a",
                              "filter": filter_name},
                    }
                },
            },
        )
        renderer = Renderer(
            load_project(root), gl_context.ctx, RenderSettings(width=64, height=64)
        )
        renderer.compile()
        try:
            return renderer.render_frame().images["image"][1, 1].copy()
        finally:
            renderer.release()

    def test_nearest_applies_to_both_references(self, make_project, gl_context):
        pixel = self._pixel(make_project, gl_context, "nearest")
        assert pixel[0] == pytest.approx(0.0)
        assert pixel[1] == pytest.approx(0.0)

    def test_linear_applies_to_both_references(self, make_project, gl_context):
        pixel = self._pixel(make_project, gl_context, "linear")
        assert pixel[0] == pytest.approx(0.5)
        assert pixel[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Config ambiguity
# --------------------------------------------------------------------------


class TestConfigAmbiguity:
    def test_two_config_files_are_rejected(self, tmp_path):
        """Preferring one silently makes edits to the other appear to do nothing."""
        root = tmp_path / "proj"
        root.mkdir()
        (root / "image.glsl").write_text(IMAGE)
        (root / "shadertoy.json").write_text('{"defaults":{"width":111}}')
        (root / "shadertoy.toml").write_text("[defaults]\nwidth = 222\n")
        with pytest.raises(ProjectError, match="multiple config files"):
            load_project(root)

    def test_dotted_and_plain_together_are_rejected(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "image.glsl").write_text(IMAGE)
        (root / "shadertoy.json").write_text("{}")
        (root / ".shadertoy.json").write_text("{}")
        with pytest.raises(ProjectError, match="multiple config files"):
            load_project(root)

    def test_one_config_file_is_fine(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "image.glsl").write_text(IMAGE)
        (root / "shadertoy.json").write_text('{"defaults":{"width":111}}')
        assert load_project(root).default("width", 0) == 111


# --------------------------------------------------------------------------
# Diagnostics through includes, and remaining uniforms
# --------------------------------------------------------------------------


@pytest.mark.gpu
class TestDiagnosticsThroughIncludes:
    def test_error_inside_an_include_maps_to_that_file(
        self, make_project, gl_context
    ):
        """The origin table must survive include expansion, not just the
        common/pass boundary."""
        from shadertoy_local.renderer import Renderer, RenderSettings, ShaderCompileError

        root = make_project(
            {
                "image.glsl": '#include "lib/util.glsl"\n'
                "void mainImage(out vec4 c, in vec2 f){ c = vec4(helper()); }\n",
                "lib/util.glsl": "// a helper\nfloat helper(){ return nope; }\n",
            }
        )
        renderer = Renderer(
            load_project(root), gl_context.ctx, RenderSettings(width=4, height=4)
        )
        try:
            with pytest.raises(ShaderCompileError) as excinfo:
                renderer.compile()
        finally:
            renderer.release()
        errors = [d for d in excinfo.value.diagnostics if d.is_error]
        assert errors[0].file == "lib/util.glsl"
        assert errors[0].line == 2


@pytest.mark.gpu
class TestRemainingUniforms:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("iChannelTime[0]", 0.5),
            ("iSampleRate / 100000.0", 0.441),
            ("iFrameRate / 100.0", 0.6),
            ("iDate.w / 100.0", 0.0),
        ],
    )
    def test_values(self, make_project, gl_context, expr, expected):
        from shadertoy_local.renderer import Renderer, RenderSettings

        source = (
            "void mainImage(out vec4 c, in vec2 f){ c = vec4(%s, 0, 0, 1); }\n" % expr
        )
        renderer = Renderer(
            load_project(make_project({"image.glsl": source})),
            gl_context.ctx,
            RenderSettings(width=4, height=4, frame=30, fps=60.0),
        )
        renderer.compile()
        try:
            value = float(renderer.render_frame().images["image"][0, 0, 0])
        finally:
            renderer.release()
        assert value == pytest.approx(expected, abs=1e-4)


@pytest.mark.gpu
class TestPrechargeOnStatelessShaders:
    @pytest.mark.parametrize("precharge", [None, 0, 5, "all"])
    def test_precharge_cannot_change_a_pure_shader(
        self, make_project, gl_context, precharge
    ):
        """A shader with no buffers is a pure function of its uniforms, so the
        warm-up window must be unobservable."""
        from shadertoy_local.renderer import Renderer, RenderSettings

        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(fract(iTime),0,0,1); }\n"
        root = make_project({"image.glsl": source})

        def render(pc):
            renderer = Renderer(
                load_project(root),
                gl_context.ctx,
                RenderSettings(width=8, height=8, frame=9, precharge=pc),
            )
            renderer.compile()
            try:
                return renderer.render_frame().images["image"].copy()
            finally:
                renderer.release()

        assert np.array_equal(render(None), render(precharge))


@pytest.mark.gpu
class TestInBetweenFramesAreRendered:
    """`--count 3 --every 20` must render frames 1..39 too, not just the three
    captured ones. Skipping them would corrupt any accumulation: the captures
    would each hold the history of only the frames actually drawn.

    The buffer here increments once per render, so its value *is* the number of
    frames rendered, which measures the timeline directly instead of trusting it.
    """

    COUNTER = (
        "void mainImage(out vec4 c, in vec2 f){\n"
        "    c = texture(iChannel0, f/iResolution.xy) + vec4(1.0);\n"
        "}\n"
    )

    def _accumulators(self, make_project, gl_context, frames, **kwargs):
        from shadertoy_local.renderer import Renderer, RenderSettings

        root = make_project(
            {"buffer_a.glsl": self.COUNTER, "image.glsl": PASSTHROUGH},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        renderer = Renderer(
            load_project(root),
            gl_context.ctx,
            RenderSettings(**{"width": 16, "height": 16, **kwargs}),
        )
        renderer.compile()
        try:
            return {
                cap.frame: round(float(cap.images["buffer_a"][0, 0, 0]))
                for cap in renderer.run(frames)
            }
        finally:
            renderer.release()

    def test_gaps_are_filled(self, make_project, gl_context):
        result = self._accumulators(make_project, gl_context, [0, 20, 40])
        assert result == {0: 1, 20: 21, 40: 41}

    def test_large_stride(self, make_project, gl_context):
        result = self._accumulators(make_project, gl_context, [0, 100])
        assert result == {0: 1, 100: 101}

    def test_precharge_trims_the_runup_but_not_the_gaps(
        self, make_project, gl_context
    ):
        """precharge governs only what precedes the first capture."""
        result = self._accumulators(
            make_project, gl_context, [20, 40, 60], precharge=5
        )
        assert result == {20: 6, 40: 26, 60: 46}
        gaps = sorted(result.values())
        assert gaps[1] - gaps[0] == 20 and gaps[2] - gaps[1] == 20

    def test_precharge_zero_still_fills_gaps(self, make_project, gl_context):
        result = self._accumulators(
            make_project, gl_context, [20, 40, 60], precharge=0
        )
        assert result == {20: 1, 40: 21, 60: 41}

    def test_unordered_capture_list_is_still_contiguous(
        self, make_project, gl_context
    ):
        result = self._accumulators(make_project, gl_context, [40, 0, 20])
        assert result == {0: 1, 20: 21, 40: 41}

    def test_duplicate_capture_frames_are_collapsed(self, make_project, gl_context):
        result = self._accumulators(make_project, gl_context, [10, 10, 10])
        assert result == {10: 11}

    def test_stateless_project_skips_the_gaps(self, make_project, gl_context):
        """With nothing to accumulate, rendering the gaps would be pure waste, so
        only the captured frames are drawn -- and the result is identical."""
        from shadertoy_local.renderer import Renderer, RenderSettings

        source = "void mainImage(out vec4 c, in vec2 f){ c = vec4(fract(iTime),0,0,1); }\n"
        root = make_project({"image.glsl": source})

        def render(frames, **kwargs):
            renderer = Renderer(
                load_project(root),
                gl_context.ctx,
                RenderSettings(**{"width": 16, "height": 16, **kwargs}),
            )
            renderer.compile()
            try:
                return {c.frame: c.images["image"].copy() for c in renderer.run(frames)}
            finally:
                renderer.release()

        skipped = render([0, 50])
        forced = render([0, 50], precharge="all")
        for frame in (0, 50):
            assert np.array_equal(skipped[frame], forced[frame]), (
                f"frame {frame} differs, so skipping the gaps was not safe"
            )


@pytest.mark.gpu
class TestCountEveryContiguityViaCli:
    def test_cli_count_every_fills_gaps(self, capsys, make_project):
        root = make_project(
            {
                "buffer_a.glsl": TestInBetweenFramesAreRendered.COUNTER,
                "image.glsl": (
                    "void mainImage(out vec4 c, in vec2 f){\n"
                    "    c = texture(iChannel0, f/iResolution.xy) / 1000.0;\n"
                    "}\n"
                ),
            },
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        code, payload, _ = _run(
            capsys, "render", "-C", str(root), "-r", "16x16",
            "--count", "3", "--every", "20", "--no-write", "--stats", "--json",
        )
        assert code == 0
        counts = [
            round(f["passes"]["image"]["stats"]["channels"]["r"]["max"] * 1000)
            for f in payload["frames"]
        ]
        assert [f["frame"] for f in payload["frames"]] == [0, 20, 40]
        assert counts == [1, 21, 41]
