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
``--no-portability``.

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
    "(see examples/06-portable-common). Disable with --no-portability."
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


# --------------------------------------------------------------------------
# Ternary on aggregate types
# --------------------------------------------------------------------------

#: Explanation for the struct-ternary check, printed once per run.
#:
#: Reported as an *error*, not a warning, and deliberately so. The Common-tab
#: check predicts cosmetic editor noise around code that still compiles and
#: renders on shadertoy.com. This one predicts an outright compile failure there,
#: so a clean local run would otherwise be actively misleading.
TERNARY_EXPLANATION = (
    "Desktop GLSL permits ?: on structs and arrays, so this compiles here, but "
    "shadertoy.com runs WebGL where it fails to compile (notably through ANGLE). "
    "Assign with if/else instead -- always valid, and no slower. Use "
    "--no-portability to build anyway."
)

_STRUCT_DECL = re.compile(r"\bstruct\s+([A-Za-z_]\w*)")


def _find_struct_names(source: str) -> set[str]:
    return set(_STRUCT_DECL.findall(source))


def _statements(source: str) -> list[tuple[int, str]]:
    """Split into ``(line_number, text)`` statements, tolerating line breaks.

    Statement-level rather than line-level so a ternary spread over several
    lines is still seen as one expression.
    """
    out: list[tuple[int, str]] = []
    buffer: list[str] = []
    line = 1
    start_line = 1
    for char in source:
        if char in ";{}":
            text = "".join(buffer).strip()
            if text:
                out.append((start_line, text))
            buffer = []
            start_line = line
            continue
        if char == "\n":
            line += 1
            if not "".join(buffer).strip():
                start_line = line
        buffer.append(char)
    text = "".join(buffer).strip()
    if text:
        out.append((start_line, text))
    return out


def lint_struct_ternary(project: Project) -> list[Diagnostic]:
    """Flag ``?:`` expressions whose result is a user-defined struct."""
    sources: list[tuple[str, str]] = []
    if project.common is not None and project.common_path is not None:
        try:
            display = str(project.common_path.relative_to(project.root))
        except ValueError:  # pragma: no cover
            display = str(project.common_path)
        sources.append((display, project.common))
    for spec in project.ordered_passes:
        try:
            display = str(spec.path.relative_to(project.root))
        except ValueError:  # pragma: no cover
            display = str(spec.path)
        sources.append((display, spec.source))

    # Structs may be declared in common and used in a pass, so collect globally.
    all_text = "\n".join(strip_comments(text) for _, text in sources)
    struct_names = _find_struct_names(all_text)
    if not struct_names:
        return []
    var_types = _struct_variable_types(all_text, struct_names)

    diagnostics: list[Diagnostic] = []
    for display, text in sources:
        cleaned = strip_comments(text)
        for line, statement in _statements(cleaned):
            if "?" not in statement:
                continue
            culprit = _struct_ternary_type(
                statement, struct_names, var_types
            )
            if culprit is None:
                continue
            diagnostics.append(
                Diagnostic(
                    # An error, not a warning: this does not compile on
                    # shadertoy.com, so passing locally would be misleading.
                    severity="error",
                    message=(
                        f"?: yields struct {culprit!r}, which does not compile on "
                        f"shadertoy.com; use if/else"
                    ),
                    pass_name="common" if display.startswith("common") else None,
                    file=display,
                    line=line,
                    code="ST-TERNARY",
                )
            )
    return diagnostics


def _struct_variable_types(source: str, struct_names: set[str]) -> dict[str, str]:
    """Map variable name -> struct type for the whole source.

    Must be whole-source: a struct ternary typically appears in ``return flag ? a
    : b``, where ``a`` and ``b`` were declared in the enclosing function's
    parameter list -- a different statement entirely.

    Function *names* are excluded: ``Ray pick(...)`` declares a function, not a
    variable, and treating ``pick`` as struct-typed would invite false positives.
    """
    types: dict[str, str] = {}
    for name in struct_names:
        for match in re.finditer(rf"\b{re.escape(name)}\s+([A-Za-z_]\w*)", source):
            tail = source[match.end() :].lstrip()
            if tail.startswith("("):
                continue
            types[match.group(1)] = name
    return types


def _split_ternary(statement: str, start: int) -> tuple[str, str] | None:
    """Given the index of a ``?``, return its two branch texts.

    Tracks bracket depth and nested ``?:`` so the matching colon is found rather
    than merely the next one.
    """
    depth = 0
    pending = 0
    for index in range(start + 1, len(statement)):
        char = statement[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if char == "?":
                pending += 1
            elif char == ":":
                if pending:
                    pending -= 1
                else:
                    return statement[start + 1 : index], statement[index + 1 :]
    return None


def _struct_ternary_type(
    statement: str, struct_names: set[str], var_types: dict[str, str]
) -> str | None:
    """Return the struct a ternary evaluates to, or None.

    Three signals are trusted, all cheap and low on false positives:

    1. the statement declares or assigns a struct-typed variable and contains
       ``?``, e.g. ``ST u = flag ? a : b``;
    2. a branch constructs a struct, e.g. ``flag ? ST(1.0) : ST(2.0)``;
    3. a branch is a *bare* struct-typed variable, e.g. ``return flag ? a : b``,
       which is how a struct ternary usually appears inside a function.

    A struct merely *mentioned* is deliberately not enough: ``flag ? s.v : 0.0``
    evaluates to a float, and flagging it would be wrong.
    """
    for name in sorted(struct_names, key=len, reverse=True):
        # (1) declaration or assignment whose target is of struct type.
        if re.search(rf"\b{re.escape(name)}\s+[A-Za-z_]\w*\s*=[^=]", statement):
            return name

    for position, char in enumerate(statement):
        if char != "?":
            continue
        branches = _split_ternary(statement, position)
        if branches is None:
            continue
        for branch in branches:
            text = branch.strip().rstrip(";").strip()
            # (2) a struct constructor.
            for name in struct_names:
                if re.match(rf"^{re.escape(name)}\s*\(", text):
                    return name
            # (3) a bare struct-typed variable (no member access or indexing).
            if re.fullmatch(r"[A-Za-z_]\w*", text) and text in var_types:
                return var_types[text]
    return None


def lint_all(project: Project) -> list[Diagnostic]:
    """Every portability check, in reporting order."""
    return [*lint_common(project), *lint_struct_ternary(project)]


#: Footers to print, keyed by the diagnostic code that triggers them.
EXPLANATIONS = {
    "ST-COMMON": EXPLANATION,
    "ST-TERNARY": TERNARY_EXPLANATION,
}
