"""Project discovery and JSON config parsing."""

from __future__ import annotations

import json
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

    def test_buffer_defaults_match_shadertoy(self, make_project, simple_image):
        """shadertoy.com defaults buffer channels to linear + clamp, no vflip.
        Diverging would make shaders render differently than on the site."""
        root = make_project(
            {"image.glsl": simple_image, "buffer_a.glsl": simple_image},
            config={"buffer_a": {"channels": {"0": "buffer_a"}}},
        )
        binding = load_project(root).passes["buffer_a"].channels[0]
        assert (binding.filter, binding.wrap, binding.vflip) == (
            "linear",
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
                            "filter": "nearest",
                            "wrap": "repeat",
                        }
                    }
                }
            },
        )
        binding = load_project(root).passes["buffer_a"].channels[0]
        assert (binding.filter, binding.wrap) == ("nearest", "repeat")

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
        with pytest.raises(ProjectError, match=r"\"filter\" must be one of"):
            load_project(root)

    def test_bad_channel_index_rejected(self, make_project, simple_image):
        root = make_project(
            {"image.glsl": simple_image},
            config={"image": {"channels": {"7": "noise"}}},
        )
        with pytest.raises(ProjectError, match="unknown key"):
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
        with pytest.raises(ProjectError, match="unknown key"):
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
        with pytest.raises(ProjectError, match=r"\"scale\" must be <= 1"):
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


class TestStrictValidation:
    """Config is validated exhaustively.

    A mistyped key that is merely ignored is the worst available outcome: the
    shader renders, at the wrong size or with the wrong sampler, and nothing says
    so. Every case here was silently accepted before.
    """

    IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"

    def _load(self, make_project, config):
        return load_project(make_project({"image.glsl": self.IMAGE}, config=config))

    def _reject(self, make_project, config, match):
        with pytest.raises(ProjectError, match=match):
            self._load(make_project, config)

    # -- defaults -------------------------------------------------------

    def test_unknown_defaults_key(self, make_project):
        self._reject(make_project, {"defaults": {"widht": 320}}, "unknown key")

    def test_unknown_key_suggests_a_near_match(self, make_project):
        self._reject(make_project, {"defaults": {"widht": 320}}, "did you mean 'width'")

    def test_defaults_must_be_an_object(self, make_project):
        self._reject(make_project, {"defaults": 5}, "must be an object")

    @pytest.mark.parametrize(
        "value,match",
        [
            ("big", "must be an integer"),
            (-5, "must be >= 1"),
            (0, "must be >= 1"),
            (True, "must be an integer"),
            (12.5, "must be an integer"),
            (99999999, "must be <= 65536"),
        ],
    )
    def test_bad_width(self, make_project, value, match):
        self._reject(make_project, {"defaults": {"width": value}}, match)

    @pytest.mark.parametrize("value,match", [(0, "must be > 0"), (-1, "must be > 0"), ("x", "must be a number")])
    def test_bad_fps(self, make_project, value, match):
        self._reject(make_project, {"defaults": {"fps": value}}, match)

    @pytest.mark.parametrize("value", [100, 999, 461, 329])
    def test_glsl_version_range(self, make_project, value):
        self._reject(make_project, {"defaults": {"glsl_version": value}}, "must be")

    def test_valid_defaults_survive(self, make_project):
        project = self._load(
            make_project,
            {"defaults": {"width": 320, "height": 180, "fps": 30, "glsl_version": 430}},
        )
        assert project.default("width", 0) == 320
        assert project.default("fps", 0) == 30.0

    # -- top level ------------------------------------------------------

    def test_name_must_be_a_string(self, make_project):
        self._reject(make_project, {"name": 123}, '"name" must be a string')

    def test_description_must_be_a_string(self, make_project):
        self._reject(make_project, {"description": []}, '"description" must be a string')

    def test_unknown_top_level_key_lists_allowed(self, make_project):
        self._reject(make_project, {"bogus": 1}, "Allowed keys:")

    # -- pass tables ----------------------------------------------------

    def test_unknown_pass_key(self, make_project):
        self._reject(make_project, {"image": {"quality": "high"}}, "unknown key")

    @pytest.mark.parametrize(
        "value,match",
        [
            (True, "must be a number"),
            ("0.5", "must be a number"),
            (0, "must be > 0"),
            (-1, "must be > 0"),
            (2, "must be <= 1"),
        ],
    )
    def test_bad_scale(self, make_project, value, match):
        self._reject(make_project, {"image": {"scale": value}}, match)

    def test_channels_must_be_an_object(self, make_project):
        self._reject(
            make_project, {"image": {"channels": ["noise"]}}, "must be an object"
        )

    # -- channel objects ------------------------------------------------

    def test_unknown_channel_key(self, make_project):
        self._reject(
            make_project,
            {"image": {"channels": {"0": {"source": "noise", "filtre": "linear"}}}},
            "did you mean 'filter'",
        )

    @pytest.mark.parametrize("alias", ["path", "texture", "file", "src"])
    def test_source_aliases_are_rejected(self, make_project, alias):
        """One spelling only: four synonyms meant a typo in one looked like a
        missing source rather than a mistake."""
        self._reject(
            make_project,
            {"image": {"channels": {"0": {alias: "noise"}}}},
            f'use "source", not "{alias}"',
        )

    @pytest.mark.parametrize("value", ["no", "yes", 0, 1, 7, None])
    def test_vflip_must_be_a_real_bool(self, make_project, value):
        self._reject(
            make_project,
            {"image": {"channels": {"0": {"source": "noise", "vflip": value}}}},
            '"vflip" must be true or false',
        )

    def test_vflip_accepts_real_bools(self, make_project):
        for value in (True, False):
            project = self._load(
                make_project,
                {"image": {"channels": {"0": {"source": "noise", "vflip": value}}}},
            )
            assert project.passes["image"].channels[0].vflip is value

    def test_wrap_typo_suggests(self, make_project):
        self._reject(
            make_project,
            {"image": {"channels": {"0": {"source": "noise", "wrap": "repaet"}}}},
            "did you mean 'repeat'",
        )

    def test_source_must_be_a_string(self, make_project):
        self._reject(
            make_project, {"image": {"channels": {"0": {"source": 5}}}},
            '"source" must be a string',
        )

    def test_empty_source(self, make_project):
        self._reject(
            make_project, {"image": {"channels": {"0": {"source": ""}}}},
            "must not be empty",
        )

    def test_missing_source(self, make_project):
        self._reject(
            make_project, {"image": {"channels": {"0": {"filter": "linear"}}}},
            'missing "source"',
        )

    def test_channel_must_not_be_a_number(self, make_project):
        self._reject(make_project, {"image": {"channels": {"0": 5}}}, "must be a string")

    @pytest.mark.parametrize("value,match", [(0, "must be >= 1"), (True, "must be an integer"), ("8", "must be an integer")])
    def test_bad_builtin_size(self, make_project, value, match):
        self._reject(
            make_project,
            {"image": {"channels": {"0": {"source": "noise", "size": value}}}},
            match,
        )

    def test_size_only_for_builtins(self, make_project):
        self._reject(
            make_project,
            {"image": {"channels": {"0": {"type": "keyboard", "size": 8}}}},
            '"size" only applies to builtin',
        )

    # -- duplicate channel spellings ------------------------------------

    def test_duplicate_channel_spelling_is_rejected(self, make_project):
        """Previously "0" silently won over "channel0"."""
        self._reject(
            make_project,
            {"image": {"channels": {"0": "noise", "channel0": "checker"}}},
            "given twice",
        )

    def test_distinct_spellings_for_distinct_channels_are_fine(self, make_project):
        project = self._load(
            make_project, {"image": {"channels": {"0": "noise", "channel1": "checker"}}}
        )
        assert project.passes["image"].channels[0].source == "noise"
        assert project.passes["image"].channels[1].source == "checker"


class TestStrictValidationTOML:
    """TOML must be validated identically; the schema is shared."""

    IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"

    def _load_toml(self, tmp_path, text):
        root = tmp_path / "proj"
        root.mkdir(exist_ok=True)
        (root / "image.glsl").write_text(self.IMAGE)
        (root / "shadertoy.toml").write_text(text, encoding="utf-8")
        return load_project(root)

    def test_defaults_typo(self, tmp_path):
        with pytest.raises(ProjectError, match="did you mean 'width'"):
            self._load_toml(tmp_path, "[defaults]\nwidht = 320\n")

    def test_bad_vflip(self, tmp_path):
        with pytest.raises(ProjectError, match='"vflip" must be true or false'):
            self._load_toml(
                tmp_path, '[image.channels.0]\nsource = "noise"\nvflip = "yes"\n'
            )

    def test_unknown_pass_key(self, tmp_path):
        with pytest.raises(ProjectError, match="unknown key"):
            self._load_toml(tmp_path, '[image]\nquality = "high"\n')

    def test_valid_toml_loads(self, tmp_path):
        project = self._load_toml(
            tmp_path,
            "[defaults]\nwidth = 320\nheight = 180\n"
            '[image.channels.0]\nsource = "noise"\nfilter = "nearest"\n',
        )
        assert project.default("width", 0) == 320
        assert project.passes["image"].channels[0].filter == "nearest"
