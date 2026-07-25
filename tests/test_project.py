"""Project discovery and JSON config parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadertoy_local.project import (
    BUFFER_NAMES,
    ProjectError,
    find_project_root,
    load_project,
)


def test_minimal_project_needs_only_image(make_project, simple_image):
    root = make_project({"image.glsl": simple_image})
    project = load_project(root)
    assert list(project.passes) == ["image"]
    assert project.config_path is None
    assert project.common is None


def test_missing_image_pass_is_an_error(tmp_path):
    (tmp_path / "buffer_a.glsl").write_text("void mainImage(out vec4 c, in vec2 f){}")
    with pytest.raises(ProjectError, match="no image pass"):
        load_project(tmp_path, search_parents=False)


def test_no_project_at_all(tmp_path):
    with pytest.raises(ProjectError, match="No Shadertoy project found"):
        find_project_root(tmp_path)


def test_discovery_walks_upward(make_project, simple_image):
    root = make_project({"image.glsl": simple_image})
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == root


def test_pass_execution_order_is_buffers_then_image(make_project, simple_image):
    files = {"image.glsl": simple_image}
    for name in BUFFER_NAMES:
        files[f"{name}.glsl"] = simple_image
    project = load_project(make_project(files))
    assert [p.name for p in project.ordered_passes] == [
        "buffer_a",
        "buffer_b",
        "buffer_c",
        "buffer_d",
        "image",
    ]


def test_common_is_discovered_by_filename(make_project, simple_image):
    root = make_project(
        {"image.glsl": simple_image, "common.glsl": "float helper() { return 1.0; }"}
    )
    project = load_project(root)
    assert project.common is not None
    assert "helper" in project.common


class TestChannelWiring:
    def test_string_shorthand_infers_type(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        binding = load_project(root).passes["image"].channels[0]
        assert binding.kind == "buffer"
        assert binding.source == "buffer_a"

    def test_builtin_and_keyboard_inference(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"0": "noise", "1": "keyboard"}}},
        )
        channels = load_project(root).passes["image"].channels
        assert channels[0].kind == "builtin"
        assert channels[1].kind == "keyboard"

    def test_keyboard_needs_no_source(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"0": {"type": "keyboard"}}}},
        )
        assert load_project(root).passes["image"].channels[0].is_keyboard

    def test_channel_index_aliases(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"channel2": "noise"}}},
        )
        assert 2 in load_project(root).passes["image"].channels

    def test_buffer_defaults_are_feedback_safe(self, make_project, simple_image):
        """Linear+repeat on a feedback buffer smears state; nearest+clamp is safer."""
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
            config={"buffer_a": {"channels": {"0": "buffer_a"}}},
        )
        binding = load_project(root).passes["buffer_a"].channels[0]
        assert (binding.filter, binding.wrap, binding.vflip) == (
            "nearest",
            "clamp",
            False,
        )

    def test_explicit_sampler_settings_win(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
            config={
                "buffer_a": {
                    "channels": {
                        "0": {
                            "source": "buffer_a",
                            "filter": "linear",
                            "wrap": "repeat",
                        }
                    }
                }
            },
        )
        binding = load_project(root).passes["buffer_a"].channels[0]
        assert (binding.filter, binding.wrap) == ("linear", "repeat")

    def test_buffer_reference_to_missing_pass(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"0": "buffer_c"}}},
        )
        with pytest.raises(ProjectError, match="no buffer_c.glsl"):
            load_project(root)

    def test_missing_texture_file(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"0": "textures/absent.png"}}},
        )
        with pytest.raises(ProjectError, match="not found"):
            load_project(root)

    def test_declared_type_mismatch_is_caught(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={
                "image": {"channels": {"0": {"type": "buffer", "source": "noise"}}}
            },
        )
        with pytest.raises(ProjectError, match="not one of"):
            load_project(root)

    def test_invalid_filter_rejected(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={
                "image": {
                    "channels": {"0": {"source": "noise", "filter": "trilinear"}}
                }
            },
        )
        with pytest.raises(ProjectError, match="filter must be one of"):
            load_project(root)

    def test_bad_channel_index_rejected(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"7": "noise"}}},
        )
        with pytest.raises(ProjectError, match="invalid key"):
            load_project(root)


class TestConfigValidation:
    def test_file_key_is_rejected_with_guidance(self, make_project, simple_image):
        """Passes are identified by filename; `file` must not silently work."""
        root = make_project(
            {"image.glsl": simple_image}, config={"image": {"file": "other.glsl"}}
        )
        with pytest.raises(ProjectError, match="identified by filename"):
            load_project(root)

    def test_unknown_top_level_key(self, make_project, simple_image):
        root = make_project({"image.glsl": simple_image}, config={"buffer_z": {}})
        with pytest.raises(ProjectError, match="unknown top-level key"):
            load_project(root)

    def test_invalid_json_reports_position(self, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        (root / "shadertoy.json").write_text('{"image": ')
        with pytest.raises(ProjectError, match="invalid JSON at line"):
            load_project(root)

    def test_scale_must_be_in_range(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
            config={"buffer_a": {"scale": 3}},
        )
        with pytest.raises(ProjectError, match=r"scale must be a number"):
            load_project(root)

    def test_toml_is_also_accepted(self, make_project, simple_image):
        root = make_project({"image.glsl": simple_image})
        (root / "shadertoy.toml").write_text(
            '[image.channels]\n0 = "noise"\n', encoding="utf-8"
        )
        project = load_project(root)
        assert project.passes["image"].channels[0].source == "noise"

    def test_defaults_are_readable(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"defaults": {"width": 111, "height": 222}},
        )
        project = load_project(root)
        assert project.default("width", 0) == 111
        assert project.default("height", 0) == 222
        assert project.default("absent", "fallback") == "fallback"


def test_to_dict_is_json_serialisable(make_project, simple_image):
    root = make_project(
        {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
        config={"image": {"channels": {"0": "buffer_a", "1": "keyboard"}}},
    )
    payload = load_project(root).to_dict()
    assert json.loads(json.dumps(payload))["passes"]["image"]["channels"]["1"][
        "type"
    ] == "keyboard"
