"""Portability linting for the Common tab.

shadertoy-local concatenates ``common.glsl`` into each pass *after* emitting the
full uniform prelude, so Common code may reference ``iTime`` and friends freely.
shadertoy.com does not behave that way: it validates the Common tab standalone
against a minimal header, so the same code lights up with ``undeclared
identifier`` diagnostics in the site's editor even though it renders correctly.

Those diagnostics are cosmetic, but they are not harmless -- roughly thirty
lines of uniform noise will camouflage a genuine typo in Common. This check
therefore runs by default (as warnings, which never change the exit code) so
Common stays clean and the site's error log stays readable. Disable it with
``--no-portable-common``.

The safe set below is **empirical**, derived from probing shadertoy.com: of the
standard uniforms, only ``iDate`` and ``iSampleRate`` survive standalone Common
validation. Those two happen to be the ones meaningful to every pass type
including Sound, which has no resolution, time, mouse or frame uniforms -- but
treat that as a plausible explanation rather than a documented contract, since
Sound passes do accept channel inputs and ``iChannel0..3`` are nonetheless
reported as undeclared. The observation is what this lint encodes; the reason is
not load-bearing.
"""

from __future__ import annotations

import re

from .compose import Origin
from .diagnostics import Diagnostic
from .project import Project

#: Uniforms that pass shadertoy.com's standalone Common validation.
COMMON_SAFE_UNIFORMS = frozenset({"iDate", "iSampleRate"})

#: Uniforms the site reports as undeclared inside the Common tab.
PASS_SPECIFIC_UNIFORMS = (
    "iResolution",
    "iTime",
    "iTimeDelta",
    "iFrameRate",
    "iFrame",
    "iChannelTime",
    "iChannelResolution",
    "iMouse",
    "iChannel0",
    "iChannel1",
    "iChannel2",
    "iChannel3",
)

_IDENTIFIER = re.compile(
    r"\b(" + "|".join(sorted(PASS_SPECIFIC_UNIFORMS, key=len, reverse=True)) + r")\b"
)

#: Printed once per run when any portability warning fires, rather than being
#: repeated on every line. This check is on by default, so it has to be quiet.
EXPLANATION = (
    "shadertoy.com validates the Common tab standalone against a minimal "
    "header, so the uniforms above are reported as undeclared in the site's "
    "editor. The shader still renders correctly there -- but that noise will "
    "camouflage genuine typos in Common. To silence it, take the value as a "
    "function parameter, or fill a uniforms struct once per pass "
    "(see examples/06-portable-common). Disable with --no-portable-common."
)


def strip_comments(source: str) -> str:
    """Blank out comments while preserving line and column positions.

    Replacing rather than deleting keeps every subsequent line number correct,
    which matters because the whole point is to report accurate positions.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    in_line_comment = False
    in_block_comment = False

    while index < length:
        char = source[index]
        pair = source[index : index + 2]

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                out.append(char)
            else:
                out.append(" ")
            index += 1
            continue

        if in_block_comment:
            if pair == "*/":
                in_block_comment = False
                out.append("  ")
                index += 2
                continue
            # Preserve newlines so line numbering survives block comments.
            out.append("\n" if char == "\n" else " ")
            index += 1
            continue

        if pair == "//":
            in_line_comment = True
            out.append("  ")
            index += 2
            continue
        if pair == "/*":
            in_block_comment = True
            out.append("  ")
            index += 2
            continue

        out.append(char)
        index += 1

    return "".join(out)


def lint_common(project: Project) -> list[Diagnostic]:
    """Warn about pass-specific uniforms referenced in ``common.glsl``."""
    if project.common is None or project.common_path is None:
        return []

    try:
        display = str(project.common_path.relative_to(project.root))
    except ValueError:  # pragma: no cover - common is always inside the project
        display = str(project.common_path)

    diagnostics: list[Diagnostic] = []
    cleaned = strip_comments(project.common)

    lines = cleaned.split("\n")
    in_directive = False
    for lineno, text in enumerate(lines, start=1):
        stripped = text.lstrip()
        # A macro body is only compiled where it is expanded, so naming a
        # pass-specific uniform inside an unexpanded #define is legal even on
        # shadertoy.com. Skipping directives keeps the macro-based uniform
        # struct pattern lint-clean, which is the whole point of using it.
        if in_directive or stripped.startswith("#"):
            in_directive = text.rstrip().endswith("\\")
            continue

        seen: set[str] = set()
        for match in _IDENTIFIER.finditer(text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    message=f"{name} is not visible in shadertoy.com's Common tab",
                    pass_name="common",
                    file=display,
                    line=lineno,
                    column=match.start() + 1,
                    code="ST-COMMON",
                )
            )
    return diagnostics


def common_origin_line(project: Project, line: int) -> Origin | None:
    """Convenience accessor used by formatters."""
    if project.common_path is None:
        return None
    try:
        display = str(project.common_path.relative_to(project.root))
    except ValueError:  # pragma: no cover
        display = str(project.common_path)
    return Origin(file=display, line=line)
