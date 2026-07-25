"""The shadertoy.com porting report.

These assertions matter because the report's whole value is being *trustworthy*:
a missed blocker sends someone to the site to discover it by hand, and a spurious
one wastes time working around a non-problem.
"""

from __future__ import annotations

import json

import pytest

from shadertoy_local.project import load_project
from shadertoy_local.wiring import build_report, format_report

IMAGE = "void mainImage(out vec4 c, in vec2 f){ c = vec4(1.0); }\n"
SAMPLE = (
    "void mainImage(out vec4 c, in vec2 f){\n"
    "    c = texture(iChannel0, f/iResolution.xy);\n"
    "}\n"
)


def report_for(make_project, files, config=None):
    return build_report(load_project(make_project(files, config=config)))


class TestTabs:
    def test_single_pass(self, make_project):
        report = report_for(make_project, {"image.glsl": IMAGE})
        assert report.tabs == [("Image", "image.glsl")]

    def test_common_is_listed_first(self, make_project):
        """A buffer must exist before anything references it, and Common before
        anything uses its helpers."""
        report = report_for(
            make_project,
            {"image.glsl": IMAGE, "common.glsl": "// x\n", "buffer_a.glsl": IMAGE},
        )
        assert [tab for tab, _ in report.tabs] == ["Common", "Buffer A", "Image"]

    def test_all_buffers(self, make_project):
        files = {"image.glsl": IMAGE}
        for letter in "abcd":
            files[f"buffer_{letter}.glsl"] = IMAGE
        report = report_for(make_project, files)
        assert [tab for tab, _ in report.tabs] == [
            "Buffer A", "Buffer B", "Buffer C", "Buffer D", "Image",
        ]


class TestChannelWiring:
    def test_buffer_maps_to_the_misc_picker(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_c.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_c"}}},
        )
        (channel,) = report.channels
        assert channel.site_input == "Misc > Buffer C"
        assert channel.note is None

    def test_keyboard_maps_to_the_misc_picker(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": {"type": "keyboard"}}}},
        )
        assert report.channels[0].site_input == "Misc > Keyboard"

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("rgba-noise-small", "Textures > RGBA Noise Small"),
            ("rgba-noise-medium", "Textures > RGBA Noise Medium"),
            ("noise", "Textures > RGBA Noise Medium"),
            ("gray-noise-small", "Textures > Gray Noise Small"),
            ("blue-noise", "Textures > Blue Noise"),
            ("bayer", "Textures > Bayer"),
        ],
    )
    def test_builtins_map_to_stock_textures(self, make_project, source, expected):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": source}}},
        )
        assert report.channels[0].site_input == expected

    def test_approximate_builtins_say_so(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": "noise"}}},
        )
        assert "different pixel values" in report.channels[0].note

    def test_bayer_is_reported_as_exact(self, make_project):
        """Unlike the noise textures, a Bayer matrix is reproducible bit for bit."""
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": "bayer"}}},
        )
        assert "exactly" in report.channels[0].note

    def test_sampler_settings_are_reported(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={
                "image": {
                    "channels": {
                        "0": {
                            "source": "noise",
                            "filter": "mipmap",
                            "wrap": "clamp",
                            "vflip": False,
                        }
                    }
                }
            },
        )
        summary = report.channels[0].sampler_summary()
        assert "filter=mipmap" in summary
        assert "wrap=clamp" in summary
        assert "vflip=off" in summary

    def test_vflip_is_omitted_for_buffers(self, make_project):
        """It has no meaning there, so showing it would imply a setting exists."""
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        assert "vflip" not in report.channels[0].sampler_summary()

    def test_channels_are_grouped_per_pass_in_order(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": SAMPLE},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a", "2": "noise"}},
            },
        )
        assert [(c.pass_name, c.index) for c in report.channels] == [
            ("Buffer A", 0),
            ("Image", 0),
            ("Image", 2),
        ]


class TestBlockers:
    def test_local_texture_file_is_a_blocker(self, make_project):
        from PIL import Image as PILImage

        root = make_project({"image.glsl": SAMPLE})
        PILImage.new("RGBA", (4, 4)).save(root / "tex.png")
        (root / "shadertoy.json").write_text(
            json.dumps(
                {"image": {"channels": {"0": {"type": "texture", "source": "tex.png"}}}}
            )
        )
        report = build_report(load_project(root))
        assert not report.portable
        assert any("no custom texture upload" in b for b in report.blockers)
        assert report.channels[0].site_input is None

    def test_local_only_builtin_is_a_blocker(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": "uv"}}},
        )
        assert any("no counterpart" in b for b in report.blockers)

    def test_buffer_scale_is_a_blocker(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": IMAGE},
            config={
                "buffer_a": {"scale": 0.5},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        assert any("always full resolution" in b for b in report.blockers)

    def test_include_is_a_blocker(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": '#include "a.glsl"\n' + IMAGE, "a.glsl": "// x\n"},
        )
        assert any("#include" in b for b in report.blockers)

    def test_include_in_common_is_also_caught(self, make_project):
        report = report_for(
            make_project,
            {
                "image.glsl": IMAGE,
                "common.glsl": '#include "a.glsl"\n',
                "a.glsl": "// x\n",
            },
        )
        assert any("#include" in b for b in report.blockers)

    def test_newer_glsl_is_a_blocker(self, make_project):
        report = report_for(
            make_project, {"image.glsl": IMAGE}, config={"defaults": {"glsl_version": 430}}
        )
        assert any("GLSL ES 3.0" in b for b in report.blockers)

    def test_default_glsl_is_not_a_blocker(self, make_project):
        report = report_for(
            make_project, {"image.glsl": IMAGE}, config={"defaults": {"glsl_version": 330}}
        )
        assert report.portable

    def test_a_plain_project_has_no_blockers(self, make_project):
        report = report_for(make_project, {"image.glsl": IMAGE})
        assert report.portable
        assert report.blockers == []

    def test_shipped_examples_report_honestly(self):
        """Everything but 04 should port cleanly; 04 deliberately cannot."""
        from .conftest import EXAMPLES_DIR

        expected_portable = {
            "01-plasma": True,
            "02-raymarch": True,
            "03-feedback-trail": True,
            "04-textured": False,
            "05-interactive": True,
            "06-portable-common": True,
            "07-path-traced-box": True,
        }
        for name, portable in expected_portable.items():
            report = build_report(load_project(EXAMPLES_DIR / name))
            assert report.portable is portable, (
                f"{name}: blockers={report.blockers}"
            )


class TestNotes:
    def test_shared_buffer_note_when_read_more_than_once(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": SAMPLE},
            config={
                "buffer_a": {"channels": {"0": "buffer_a"}},
                "image": {"channels": {"0": "buffer_a"}},
            },
        )
        assert any("property of the buffer" in n for n in report.notes)

    def test_no_shared_buffer_note_for_a_single_reader(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        assert not any("property of the buffer" in n for n in report.notes)

    def test_no_generic_boilerplate_notes(self, make_project):
        """Notes must be project-specific and actionable. The Common-tab caveat is
        already reported per-line by `check`, and a viewport/resolution note
        applies to every project alike, so neither belongs here."""
        report = report_for(
            make_project, {"image.glsl": IMAGE, "common.glsl": "// x\n"}
        )
        assert report.notes == []

    def test_local_only_builtin_has_no_duplicate_note(self, make_project):
        """It is already a blocker; saying it twice is noise."""
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": "uv"}}},
        )
        assert report.blockers
        assert report.notes == []


class TestFormatting:
    def test_lines_render_without_error(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a", "1": "noise"}}},
        )
        lines = format_report(report)
        text = "\n".join(lines)
        assert "Tabs to create" in text
        assert "iChannel0" in text and "Misc > Buffer A" in text

    def test_no_inputs_is_stated_explicitly(self, make_project):
        lines = format_report(report_for(make_project, {"image.glsl": IMAGE}))
        assert any("no inputs to configure" in line for line in lines)

    def test_blockers_are_rendered(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE},
            config={"image": {"channels": {"0": "checker"}}},
        )
        text = "\n".join(format_report(report))
        assert "Cannot be reproduced as-is" in text

    def test_worth_knowing_section_is_omitted_when_empty(self, make_project):
        lines = format_report(report_for(make_project, {"image.glsl": IMAGE}))
        assert not any("Worth knowing" in line for line in lines)

    def test_report_is_serialisable(self, make_project):
        report = report_for(
            make_project,
            {"image.glsl": SAMPLE, "buffer_a.glsl": IMAGE},
            config={"image": {"channels": {"0": "buffer_a"}}},
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["portable"] is True
        assert payload["channels"][0]["site_input"] == "Misc > Buffer A"


class TestSingleSourceOfTruth:
    """The file summary and the tab list are both derived from
    ``Project.ordered_files``, so they cannot disagree about which files exist.
    They previously did: Common is not a pass, so it never appeared in
    ``ordered_passes``, which the summary iterated while the tab list built its
    own sequence.
    """

    def test_tabs_are_exactly_the_project_files(self, make_project):
        project = load_project(
            make_project(
                {
                    "common.glsl": "// x\n",
                    "buffer_a.glsl": IMAGE,
                    "buffer_c.glsl": IMAGE,
                    "image.glsl": IMAGE,
                }
            )
        )
        report = build_report(project)
        assert report.tabs == [
            (entry.label, entry.path.name) for entry in project.ordered_files
        ]

    def test_ordered_files_places_common_first(self, make_project):
        project = load_project(
            make_project(
                {"common.glsl": "// x\n", "buffer_b.glsl": IMAGE, "image.glsl": IMAGE}
            )
        )
        assert [e.label for e in project.ordered_files] == [
            "Common",
            "Buffer B",
            "Image",
        ]

    def test_common_is_not_a_pass(self, make_project):
        project = load_project(
            make_project({"common.glsl": "// x\n", "image.glsl": IMAGE})
        )
        by_label = {e.label: e for e in project.ordered_files}
        assert by_label["Common"].is_pass is False
        assert by_label["Common"].spec is None
        assert by_label["Image"].is_pass is True

    def test_ordered_files_without_common(self, make_project):
        project = load_project(make_project({"image.glsl": IMAGE}))
        assert [e.label for e in project.ordered_files] == ["Image"]

    def test_source_files_is_derived_from_ordered_files(self, make_project):
        project = load_project(
            make_project({"common.glsl": "// x\n", "image.glsl": IMAGE})
        )
        assert project.source_files() == [e.path for e in project.ordered_files]

    def test_labels_match_the_pass_labels(self, make_project):
        """wiring no longer keeps its own tab-name table; a second mapping would
        be free to drift from PassSpec.label."""
        project = load_project(
            make_project({"buffer_d.glsl": IMAGE, "image.glsl": IMAGE})
        )
        labels = {e.label for e in project.ordered_files if e.spec is not None}
        assert labels == {spec.label for spec in project.ordered_passes}

    @pytest.mark.gpu
    def test_cli_summary_and_tab_list_agree(self, capsys, make_project):
        """The end-to-end property that was actually broken."""
        from shadertoy_local.cli import main

        root = make_project(
            {"common.glsl": "// x\n", "buffer_a.glsl": IMAGE, "image.glsl": IMAGE}
        )
        main(["info", "-C", str(root), "--no-runtime-check"])
        text = capsys.readouterr().err
        summary, _, porting = text.partition("-- porting to shadertoy.com")
        for filename in ("common.glsl", "buffer_a.glsl", "image.glsl"):
            assert filename in summary, f"file summary omitted {filename}"
            assert filename in porting, f"tab list omitted {filename}"
