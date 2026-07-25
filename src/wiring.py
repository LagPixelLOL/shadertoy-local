"""Porting report: what to do on shadertoy.com to reproduce a local project.

``shadertoy info`` used to list which files exist, which is not what you need when
sitting in front of the real site. What you need is which tabs to create, what to
select in each ``iChannel`` slot, which sampler settings to set, and -- most
usefully -- which parts of the project *cannot* be reproduced there at all.

That last category is the point. A local project can express things the site
cannot: buffers at reduced resolution, ``#include``, procedural textures with no
stock equivalent, and GLSL newer than ES 3.0. Discovering those one at a time while
clicking around the site is miserable, so they are listed up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compose import DEFAULT_GLSL_VERSION
from .project import BUFFER_NAMES, ChannelBinding, Project

#: Local builtin -> where to find its counterpart in shadertoy.com's input picker.
#: Only the ones that genuinely have a counterpart appear here.
_BUILTIN_TO_SITE: dict[str, str] = {
    "rgba-noise-small": "Textures > RGBA Noise Small",
    "rgba-noise-medium": "Textures > RGBA Noise Medium",
    "rgba-noise": "Textures > RGBA Noise Medium",
    "noise": "Textures > RGBA Noise Medium",
    "gray-noise-small": "Textures > Gray Noise Small",
    "gray-noise-medium": "Textures > Gray Noise Medium",
    "gray-noise": "Textures > Gray Noise Medium",
    "blue-noise": "Textures > Blue Noise",
    "bayer": "Textures > Bayer",
}

#: Builtins whose pixels match the site's asset exactly. A Bayer matrix is
#: defined by recurrence rather than authored, so it can be reproduced bit for bit.
_EXACT_BUILTINS = frozenset({"bayer"})

#: How a pass name appears as a tab on the site.
_TAB_NAMES = {
    "image": "Image",
    "buffer_a": "Buffer A",
    "buffer_b": "Buffer B",
    "buffer_c": "Buffer C",
    "buffer_d": "Buffer D",
}


@dataclass
class ChannelWiring:
    """One ``iChannelN`` slot and what to put in it."""

    pass_name: str
    index: int
    #: What to pick in the site's input picker, or None when nothing matches.
    site_input: str | None
    source: str
    kind: str
    filter: str
    wrap: str
    vflip: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_name,
            "channel": self.index,
            "site_input": self.site_input,
            "source": self.source,
            "type": self.kind,
            "filter": self.filter,
            "wrap": self.wrap,
            "vflip": self.vflip,
            "note": self.note,
        }

    def sampler_summary(self) -> str:
        parts = [f"filter={self.filter}", f"wrap={self.wrap}"]
        # vflip is only meaningful for image-backed inputs.
        if self.kind in ("texture", "builtin"):
            parts.append(f"vflip={'on' if self.vflip else 'off'}")
        return "  ".join(parts)


@dataclass
class PortingReport:
    """Everything needed to recreate a project on shadertoy.com."""

    #: Tab name -> local filename, in the order they should be created.
    tabs: list[tuple[str, str]] = field(default_factory=list)
    channels: list[ChannelWiring] = field(default_factory=list)
    #: Things the site cannot reproduce, or that need manual attention.
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def portable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "portable": self.portable,
            "tabs": [{"tab": tab, "file": path} for tab, path in self.tabs],
            "channels": [c.to_dict() for c in self.channels],
            "blockers": self.blockers,
            "notes": self.notes,
        }


def _describe_binding(binding: ChannelBinding) -> tuple[str | None, str | None]:
    """Return ``(site_input, note)`` for one binding."""
    if binding.is_keyboard:
        return "Misc > Keyboard", None
    if binding.is_buffer:
        letter = binding.source.rsplit("_", 1)[1].upper()
        return f"Misc > Buffer {letter}", None
    if binding.kind == "builtin":
        site = _BUILTIN_TO_SITE.get(binding.source)
        if site is not None:
            if binding.source in _EXACT_BUILTINS:
                return site, "matches the site's asset exactly"
            return site, "same role and size, but different pixel values"
        return None, (
            f"builtin {binding.source!r} has no counterpart on shadertoy.com"
        )
    # A file texture: the site has no custom-image upload for shader inputs.
    return None, (
        f"{binding.source} is a local file; shadertoy.com has no custom texture "
        f"upload, so substitute a stock texture or generate it in code"
    )


def build_report(project: Project) -> PortingReport:
    """Work out how to reproduce *project* on shadertoy.com."""
    report = PortingReport()

    # Tabs, in an order where a buffer exists before anything references it.
    if project.common_path is not None:
        report.tabs.append(("Common", project.common_path.name))
    for spec in project.ordered_passes:
        report.tabs.append((_TAB_NAMES[spec.name], spec.path.name))

    buffer_users: dict[str, list[str]] = {}

    for spec in project.ordered_passes:
        for index, binding in sorted(spec.channels.items()):
            site_input, note = _describe_binding(binding)
            report.channels.append(
                ChannelWiring(
                    pass_name=_TAB_NAMES[spec.name],
                    index=index,
                    site_input=site_input,
                    source=binding.source,
                    kind=binding.kind,
                    filter=binding.filter,
                    wrap=binding.wrap,
                    vflip=binding.vflip,
                    note=note,
                )
            )
            if site_input is None and note:
                report.blockers.append(f"{_TAB_NAMES[spec.name]} iChannel{index}: {note}")
            if binding.is_buffer:
                buffer_users.setdefault(binding.source, []).append(
                    f"{_TAB_NAMES[spec.name]} iChannel{index}"
                )

    # -- things the site cannot express ---------------------------------

    for spec in project.ordered_passes:
        if spec.name in BUFFER_NAMES and spec.scale != 1.0:
            report.blockers.append(
                f"{_TAB_NAMES[spec.name]} uses scale={spec.scale}; buffers on "
                f"shadertoy.com are always full resolution"
            )

    for path in project.source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover
            continue
        if "#include" in text:
            try:
                shown = path.relative_to(project.root)
            except ValueError:  # pragma: no cover
                shown = path
            report.blockers.append(
                f"{shown} uses #include; shadertoy.com has no include directive, "
                f"so inline it before pasting"
            )

    version = int(project.default("glsl_version", DEFAULT_GLSL_VERSION))
    if version > DEFAULT_GLSL_VERSION:
        report.blockers.append(
            f"glsl_version is {version}; shadertoy.com is GLSL ES 3.0, roughly "
            f"equivalent to {DEFAULT_GLSL_VERSION}"
        )

    # -- advisory notes -------------------------------------------------

    for name, users in sorted(buffer_users.items()):
        if len(users) > 1:
            letter = name.rsplit("_", 1)[1].upper()
            report.notes.append(
                f"Buffer {letter} is read by {len(users)} channels "
                f"({', '.join(users)}); its filter and wrap are a property of the "
                f"buffer on the site, so set them once"
            )

    return report


def format_report(report: PortingReport, *, indent: str = "  ") -> list[str]:
    """Render the report as human-readable lines."""
    lines: list[str] = []

    lines.append(f"{indent}Tabs to create on shadertoy.com:")
    width = max((len(tab) for tab, _ in report.tabs), default=0)
    for tab, filename in report.tabs:
        lines.append(f"{indent}  {tab:<{width}}  {filename}")

    if report.channels:
        lines.append("")
        lines.append(f"{indent}Channel wiring:")
        current = None
        for channel in report.channels:
            if channel.pass_name != current:
                current = channel.pass_name
                lines.append(f"{indent}  {current}")
            target = channel.site_input or f"NO EQUIVALENT ({channel.source})"
            lines.append(
                f"{indent}    iChannel{channel.index}  {target}"
            )
            lines.append(f"{indent}              {channel.sampler_summary()}")
            if channel.note:
                lines.append(f"{indent}              note: {channel.note}")
    else:
        lines.append("")
        lines.append(f"{indent}Channel wiring: no inputs to configure")

    if report.blockers:
        lines.append("")
        lines.append(f"{indent}Cannot be reproduced as-is:")
        for blocker in report.blockers:
            lines.append(f"{indent}  - {blocker}")

    if report.notes:
        lines.append("")
        lines.append(f"{indent}Worth knowing:")
        for note in report.notes:
            lines.append(f"{indent}  - {note}")

    return lines
