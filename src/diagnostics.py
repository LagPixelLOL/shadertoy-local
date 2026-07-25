"""Driver diagnostic parsing and remapping.

GLSL compilers report errors against the *composed* translation unit, which
includes ~27 lines of generated prelude. Reporting those raw numbers forces
whoever is reading (human or agent) to mentally subtract an offset that changes
whenever the prelude does. This module parses the driver log, remaps every
position through :class:`~shadertoy_local.compose.ComposedShader`, and renders
it as ``file:line`` against the source actually on disk.

Log formats differ per driver, so all the common ones are handled:

* NVIDIA  ``0(28) : error C1503: undefined variable "foo"``
* Mesa    ``0:28(10): error: no matching function for call to ...``
* AMD     ``ERROR: 0:28: 'foo' : undeclared identifier``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .compose import ComposedShader

_PATTERNS = (
    # Mesa / Intel: 0:28(10): error: message
    re.compile(
        r"^\s*(?P<src>\d+):(?P<line>\d+)\((?P<col>\d+)\)\s*:\s*"
        r"(?P<sev>error|warning|note)\s*:\s*(?P<msg>.*)$",
        re.IGNORECASE,
    ),
    # NVIDIA: 0(28) : error C1503: message
    re.compile(
        r"^\s*(?P<src>\d+)\((?P<line>\d+)\)\s*:\s*(?P<sev>error|warning|note)\s*"
        r"(?P<code>[A-Za-z]\d+)?\s*:\s*(?P<msg>.*)$",
        re.IGNORECASE,
    ),
    # AMD / reference compiler: ERROR: 0:28: message
    re.compile(
        r"^\s*(?P<sev>error|warning)\s*:\s*(?P<src>\d+):(?P<line>\d+)\s*:\s*"
        r"(?P<msg>.*)$",
        re.IGNORECASE,
    ),
    # Positionless: ERROR: message
    re.compile(r"^\s*(?P<sev>error|warning)\s*:\s*(?P<msg>.*)$", re.IGNORECASE),
)

#: moderngl decorates the driver log; these lines carry no information.
_NOISE = re.compile(
    r"^\s*(=+|-+|GLSL Compiler failed|GLSL Linker failed|"
    r"vertex_shader|fragment_shader|geometry_shader|compute_shader|"
    r"Program linking failed)\s*$",
    re.IGNORECASE,
)


@dataclass
class Diagnostic:
    """One compiler message, remapped to user source where possible."""

    severity: str
    message: str
    pass_name: str | None = None
    #: Position in the user's file (``None`` when it fell in generated code).
    file: str | None = None
    line: int | None = None
    column: int | None = None
    #: Position in the composed translation unit.
    composed_line: int | None = None
    code: str | None = None
    #: True when the position was inferred rather than mapped exactly.
    approximate: bool = False

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def location(self) -> str:
        if self.file and self.line:
            base = f"{self.file}:{self.line}"
            if self.column:
                base += f":{self.column}"
            return base + (" (approx)" if self.approximate else "")
        if self.composed_line:
            return f"<generated>:{self.composed_line}"
        return "<unknown>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "pass": self.pass_name,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "composed_line": self.composed_line,
            "code": self.code,
            "approximate": self.approximate,
        }


def parse_log(
    raw: str, composed: ComposedShader | None = None, pass_name: str | None = None
) -> list[Diagnostic]:
    """Parse a driver log into diagnostics, remapping positions via *composed*."""
    diagnostics: list[Diagnostic] = []
    seen: set[tuple] = set()

    for raw_line in raw.splitlines():
        text = raw_line.strip()
        if not text or _NOISE.match(text):
            continue

        for pattern in _PATTERNS:
            match = pattern.match(text)
            if match is None:
                continue
            groups = match.groupdict()
            severity = groups["sev"].lower()
            if severity not in ("error", "warning", "note"):
                severity = "error"
            composed_line = int(groups["line"]) if groups.get("line") else None
            diag = Diagnostic(
                severity=severity,
                message=groups["msg"].strip(),
                pass_name=pass_name,
                composed_line=composed_line,
                column=int(groups["col"]) if groups.get("col") else None,
                code=groups.get("code"),
            )
            if composed_line is not None and composed is not None:
                origin = composed.origin_of(composed_line)
                if origin is None:
                    origin = composed.nearest_origin(composed_line)
                    diag.approximate = origin is not None
                if origin is not None:
                    diag.file = origin.file
                    diag.line = origin.line
            key = (diag.severity, diag.message, diag.file, diag.line, diag.composed_line)
            if key not in seen:
                seen.add(key)
                diagnostics.append(diag)
            break
        else:
            # Unrecognised but non-empty: surface it rather than swallow it.
            key = ("error", text, None, None, None)
            if key not in seen:
                seen.add(key)
                diagnostics.append(
                    Diagnostic(severity="error", message=text, pass_name=pass_name)
                )

    return diagnostics


def format_diagnostics(
    diagnostics: Iterable[Diagnostic],
    composed: ComposedShader | None = None,
    *,
    context: int = 1,
    color: bool = False,
) -> str:
    """Render diagnostics with a source snippet and caret."""
    diagnostics = list(diagnostics)
    if not diagnostics:
        return ""

    lines = composed.source.splitlines() if composed else []
    bold = "\033[1m" if color else ""
    red = "\033[31m" if color else ""
    yellow = "\033[33m" if color else ""
    dim = "\033[2m" if color else ""
    reset = "\033[0m" if color else ""

    out: list[str] = []
    for diag in diagnostics:
        tint = red if diag.is_error else yellow
        label = f"{diag.severity}"
        if diag.code:
            label += f" [{diag.code}]"
        prefix = f"{bold}{diag.location()}{reset}: {tint}{label}{reset}: "
        out.append(prefix + diag.message)

        if diag.composed_line and lines:
            start = max(1, diag.composed_line - context)
            end = min(len(lines), diag.composed_line + context)
            width = len(str(end))
            for num in range(start, end + 1):
                marker = ">" if num == diag.composed_line else " "
                origin = composed.origin_of(num) if composed else None
                shown = origin.line if origin else num
                body = lines[num - 1]
                if num == diag.composed_line:
                    out.append(f"  {marker} {shown:>{width}} | {body}")
                    if diag.column:
                        pad = " " * (diag.column - 1)
                        out.append(
                            f"    {' ' * width} | {pad}{tint}^{reset}"
                        )
                else:
                    out.append(f"  {dim}{marker} {shown:>{width}} | {body}{reset}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def summarize(diagnostics: Iterable[Diagnostic]) -> tuple[int, int]:
    """Return ``(error_count, warning_count)``."""
    diagnostics = list(diagnostics)
    errors = sum(1 for d in diagnostics if d.severity == "error")
    warnings = sum(1 for d in diagnostics if d.severity == "warning")
    return errors, warnings
