"""Consistency guards against duplicated knowledge.

Every assertion here compares two places that must agree about one fact. None of
them test behaviour: they exist because a fact stated twice will eventually be
changed in one place only, and the resulting bug is invisible to tests that
exercise each place in isolation -- both remain individually "correct" while
contradicting each other.

That is not hypothetical here. The CLI file summary and the porting guide each
derived the project's file list independently, so Common appeared in one and not
the other; two constants listed the local-only builtins; and the mapping from
filter names to GL modes was written twice with an ``else: linear`` fallback that
would silently absorb any newly declared filter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import shadertoy_local.channels as channels
import shadertoy_local.compose as compose
import shadertoy_local.context as context
import shadertoy_local.portability as portability
import shadertoy_local.project as project
import shadertoy_local.wiring as wiring

SRC = Path(__file__).resolve().parent.parent / "src"


def _sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in SRC.glob("*.py")}


class TestBuiltinTextureLists:
    def test_declared_builtins_all_have_generators(self):
        """A builtin accepted by config validation but unknown to the generator
        would pass `check` and then fail at render. Aliases resolve to a
        canonical name first; every canonical name needs a generator and a
        default size, and every generator must be reachable from the declared
        list."""
        declared = {
            project.canonical_builtin(name)
            for name in project.BUILTIN_TEXTURES
            if name != "keyboard"
        }
        assert declared == set(channels._GENERATORS)
        assert declared == set(project.BUILTIN_DEFAULT_SIZES)

    def test_aliases_resolve_to_declared_builtins(self):
        for alias, target in project.BUILTIN_ALIASES.items():
            assert alias in project.BUILTIN_TEXTURES
            assert target in project.BUILTIN_TEXTURES
            assert target not in project.BUILTIN_ALIASES, "aliases must not chain"

    def test_builtin_texture_groups_partition_the_whole_list(self):
        assert set(project.BUILTIN_TEXTURES) == (
            set(project.SHADERTOY_LIKE_BUILTINS)
            | set(project.LOCAL_ONLY_BUILTINS)
            | {"keyboard"}
        )

    def test_the_two_groups_do_not_overlap(self):
        assert not (
            set(project.SHADERTOY_LIKE_BUILTINS) & set(project.LOCAL_ONLY_BUILTINS)
        )

    def test_local_only_builtins_are_defined_once(self):
        """channels.py used to carry an unused copy of this list."""
        holders = [
            name
            for name, module in (
                ("project", project),
                ("channels", channels),
                ("wiring", wiring),
            )
            if hasattr(module, "LOCAL_ONLY_BUILTINS")
        ]
        assert holders == ["project"]

    def test_every_generator_produces_a_usable_array(self):
        for name in channels._GENERATORS:
            array = channels.builtin_array(name, 8)
            assert array.shape == (8, 8, 4), name

    def test_every_builtin_has_a_default_size(self):
        for name in channels._GENERATORS:
            assert channels.builtin_default_size(name) is not None, name


class TestSiteMapping:
    def test_shadertoy_like_builtins_are_all_mapped(self):
        """Claiming a builtin mirrors a site asset while not knowing where to find
        it on the site is a contradiction."""
        assert set(project.SHADERTOY_LIKE_BUILTINS) == set(wiring._BUILTIN_TO_SITE)

    def test_local_only_builtins_are_never_mapped(self):
        assert not (set(project.LOCAL_ONLY_BUILTINS) & set(wiring._BUILTIN_TO_SITE))

    def test_exact_builtins_are_a_subset_of_mapped_ones(self):
        assert set(wiring._EXACT_BUILTINS) <= set(wiring._BUILTIN_TO_SITE)


class TestSamplerModeTranslation:
    def test_every_declared_filter_translates(self):
        """The old nearest/mipmap/else-linear form silently absorbed anything new."""

        class FakeCtx:
            NEAREST = "nearest"
            LINEAR = "linear"
            LINEAR_MIPMAP_LINEAR = "mipmap"

        for name in project._FILTERS:
            modes = channels.sampler_filter_modes(FakeCtx(), name)
            assert isinstance(modes, tuple) and len(modes) == 2, name

    def test_unknown_filter_raises_rather_than_defaulting(self):
        class FakeCtx:
            NEAREST = LINEAR = LINEAR_MIPMAP_LINEAR = 0

        with pytest.raises(project.ProjectError, match="unsupported filter"):
            channels.sampler_filter_modes(FakeCtx(), "trilinear")

    def test_every_declared_wrap_translates(self):
        for name in project._WRAPS:
            assert isinstance(channels.sampler_repeats(name), bool), name

    def test_unknown_wrap_raises(self):
        with pytest.raises(project.ProjectError, match="unsupported wrap"):
            channels.sampler_repeats("mirror")

    def test_filters_are_translated_in_exactly_one_place(self):
        """Two translation sites is how the fallback bug arose."""
        offenders = [
            name
            for name, text in _sources().items()
            if "LINEAR_MIPMAP_LINEAR" in text and name != "channels.py"
        ]
        assert offenders == []


class TestDefaultsSchema:
    def test_every_default_read_is_settable(self):
        """Reading a default that the schema rejects means the config cannot set
        it, so the value is silently frozen at its fallback."""
        read: set[str] = set()
        for text in _sources().values():
            read |= set(re.findall(r'\.default\(\s*"([a-z_]+)"', text))
        assert read <= set(project._DEFAULTS_SCHEMA), sorted(
            read - set(project._DEFAULTS_SCHEMA)
        )

    def test_every_settable_default_is_read(self):
        """The converse: a schema key nothing reads is a promise never kept."""
        read: set[str] = set()
        for text in _sources().values():
            read |= set(re.findall(r'\.default\(\s*"([a-z_]+)"', text))
        assert set(project._DEFAULTS_SCHEMA) <= read, sorted(
            set(project._DEFAULTS_SCHEMA) - read
        )


class TestPassTables:
    def test_pass_names_match_the_filename_table(self):
        assert set(project.PASS_NAMES) == set(project._PASS_FILENAMES)

    def test_buffer_names_are_a_subset_of_pass_names(self):
        assert set(project.BUFFER_NAMES) < set(project.PASS_NAMES)

    def test_image_is_last_so_it_renders_after_every_buffer(self):
        assert project.PASS_NAMES[-1] == "image"

    def test_every_pass_has_a_distinct_label(self):
        labels = [
            project.PassSpec(name=name, path=Path("x"), source="").label
            for name in project.PASS_NAMES
        ]
        assert len(set(labels)) == len(labels)

    def test_pass_labels_are_not_duplicated_as_literals(self):
        """wiring.py used to keep its own pass-name-to-tab-name table."""
        offenders = [
            name
            for name, text in _sources().items()
            if '"Buffer A"' in text or "'Buffer A'" in text
        ]
        assert offenders == []


class TestGlslVersionConstants:
    def test_no_hardcoded_fallback_beside_the_constant(self):
        literals = set()
        for text in _sources().values():
            literals |= set(re.findall(r'default\("glsl_version",\s*(\d+)\)', text))
        assert all(int(v) == compose.DEFAULT_GLSL_VERSION for v in literals), literals

    def test_schema_range_admits_every_attempted_version(self):
        """create_context walks a ladder of versions; the config must be able to
        pin any of them."""
        validator = project._DEFAULTS_SCHEMA["glsl_version"]
        for version in context._VERSION_LADDER:
            assert validator(version, "test") == version

    def test_default_version_is_admitted_by_the_schema(self):
        validator = project._DEFAULTS_SCHEMA["glsl_version"]
        assert validator(compose.DEFAULT_GLSL_VERSION, "test") == (
            compose.DEFAULT_GLSL_VERSION
        )


class TestPortabilityCodes:
    def test_every_emitted_code_has_an_explanation(self):
        emitted = set(
            re.findall(r'code="(ST-[A-Z]+)"', _sources()["portability.py"])
        )
        assert emitted, "no diagnostic codes found; the regex has rotted"
        assert emitted <= set(portability.EXPLANATIONS)

    def test_every_explanation_belongs_to_an_emitted_code(self):
        emitted = set(
            re.findall(r'code="(ST-[A-Z]+)"', _sources()["portability.py"])
        )
        assert set(portability.EXPLANATIONS) <= emitted

    def test_pass_specific_uniforms_exclude_the_safe_ones(self):
        assert not (
            set(portability.PASS_SPECIFIC_UNIFORMS) & portability.COMMON_SAFE_UNIFORMS
        )

    def test_every_prelude_uniform_is_classified(self):
        """A uniform in the prelude that neither list mentions would never be
        checked for Common-tab visibility."""
        prelude = compose._prelude(compose.DEFAULT_GLSL_VERSION)
        declared = set(re.findall(r"uniform\s+\w+\s+(i[A-Za-z0-9]+)", prelude))
        classified = set(portability.PASS_SPECIFIC_UNIFORMS) | set(
            portability.COMMON_SAFE_UNIFORMS
        )
        assert declared <= classified, sorted(declared - classified)


class TestChannelTypeHandling:
    def test_every_channel_type_is_inferable(self):
        samples = {
            "buffer": "buffer_a",
            "builtin": "noise",
            "keyboard": "keyboard",
            "texture": "some/file.png",
        }
        assert set(samples) == set(project.CHANNEL_TYPES)
        for kind, source in samples.items():
            assert project._infer_kind(source) == kind

    def test_channel_keys_cover_every_binding_field(self):
        """A ChannelBinding field with no config key is unreachable; a config key
        with no field is silently dropped."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(project.ChannelBinding)}
        # `kind` is spelled `type` in config; `path` is derived, not written.
        expected = (set(project._CHANNEL_KEYS) - {"type"}) | {"kind", "path"}
        assert expected == fields, (expected ^ fields)
