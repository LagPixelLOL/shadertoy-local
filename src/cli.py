"""Command-line interface.

Conventions that make this usable from a script or agent:

* Every command accepts ``--json`` and prints a single machine-readable object
  on stdout. Human text goes to stderr, so ``--json`` output is never polluted.
* Exit codes are meaningful: ``0`` success, ``1`` shader/assertion failure,
  ``2`` bad usage or project error, ``3`` environment/GPU unavailable.
* Nothing depends on wall-clock time, so identical arguments give identical
  pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3


class _Reporter:
    """Routes human text to stderr and the machine payload to stdout."""

    def __init__(self, as_json: bool, quiet: bool = False) -> None:
        self.as_json = as_json
        self.quiet = quiet

    def say(self, text: str = "") -> None:
        """Informational prose; hidden by --quiet and --json."""
        if not self.quiet and not self.as_json:
            print(text, file=sys.stderr)

    def warn(self, text: str) -> None:
        """Errors and failures; always shown, so --json still explains itself."""
        print(text, file=sys.stderr)

    def emit(self, payload: dict[str, Any]) -> None:
        if self.as_json:
            json.dump(payload, sys.stdout, indent=2, sort_keys=False)
            sys.stdout.write("\n")


def _fmt(value: float | None, spec: str = ".6g") -> str:
    """Format a number that may be None (an all-NaN channel has no min/max)."""
    return "n/a" if value is None else format(value, spec)


# --------------------------------------------------------------------------
# Shared argument groups
# --------------------------------------------------------------------------


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-C",
        "--directory",
        default=".",
        metavar="DIR",
        help="project directory (default: current, searched upward)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit a machine-readable JSON report"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress prose")


def _add_gpu_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("gpu")
    group.add_argument(
        "--device", type=int, metavar="N", help="EGL device index (default: auto)"
    )
    group.add_argument(
        "--backend", default=None, help="moderngl backend (default: egl)"
    )
    group.add_argument(
        "--allow-software",
        action="store_true",
        help="permit rendering on a CPU rasterizer",
    )
    group.add_argument(
        "--gl-version",
        type=int,
        default=None,
        metavar="CODE",
        help="require an exact GL version code, e.g. 430",
    )


def _add_frame_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frame")
    group.add_argument("-w", "--width", type=int, default=None, help="output width")
    group.add_argument("-H", "--height", type=int, default=None, help="output height")
    group.add_argument(
        "-r",
        "--resolution",
        default=None,
        metavar="WxH",
        help="output resolution, e.g. 640x360",
    )
    group.add_argument(
        "-f",
        "--frame",
        type=int,
        default=None,
        help="frame index to render (default: 0)",
    )
    group.add_argument("--fps", type=float, default=None, help="frame rate (default: 60)")
    group.add_argument(
        "--precharge",
        default=None,
        metavar="N|all",
        help=(
            "warm-up frames rendered before the first captured frame, giving "
            "feedback buffers history. A count, or 'all' to start from frame 0. "
            "Default: 'all' when the project has buffer passes, otherwise 0, "
            "since a shader with no buffers accumulates nothing"
        ),
    )
    group.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="abort if the run would render more than N frames (default: 100000)",
    )
    group.add_argument(
        "--date",
        default=None,
        metavar="Y,M,D,S",
        help="override iDate (default: fixed, for reproducibility)",
    )


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("simulated input")
    group.add_argument(
        "--input",
        dest="input_spec",
        default=None,
        metavar="SPEC",
        help=(
            "simulated pointer and keyboard input: a JSON array of operations, "
            "given inline, as a file path, or '-' for stdin. Each operation takes "
            "a \"frame\" or \"time\" plus an \"op\" of "
            "mouse_down/mouse_up/mouse_move/key_down/key_up/key_tap/"
            "key_toggle/key_untoggle. See --help-input."
        ),
    )
    group.add_argument(
        "--help-input",
        action="store_true",
        help="print the input operation format with examples, then exit",
    )


INPUT_HELP = """\
Simulated input is one ordered list of operations covering pointer and keyboard
together. Pass it with --input, as inline JSON, a file path, or '-' for stdin.

Each operation needs:

  "frame": N   or   "time": SECONDS     when it happens (time uses --fps)
  "op": ...                             what happens

Operations:

  mouse_down    press the button; takes an optional "pos"
  mouse_up      release the button; takes an optional "pos"
  mouse_move    move the cursor; requires "pos"
  key_down      hold keys until a matching key_up; requires "keys"
  key_up        release keys; requires "keys"
  key_tap       hold keys for exactly this one frame; requires "keys"
  key_toggle    flip the toggle row for keys; requires "keys"
  key_untoggle  clear the toggle row for keys; requires "keys"

  "pos": [x, y]           pixels, or fractions with "normalized": true
  "keys": ["w", "space"]  names or numeric JavaScript key codes

Example -- drag from the centre to the right while holding W, tap space at 1s:

  [
    {"frame": 0,  "op": "mouse_down", "pos": [320, 180]},
    {"frame": 0,  "op": "key_down",   "keys": ["w"]},
    {"frame": 30, "op": "mouse_move", "pos": [500, 180]},
    {"time": 1.0, "op": "key_tap",    "keys": ["space"]},
    {"frame": 60, "op": "mouse_up"},
    {"frame": 60, "op": "key_up",     "keys": ["w"]}
  ]

In the shader:

  iMouse.xy               cursor in pixels
  iMouse.z > 0.0          button held
  iMouse.w > 0.0          this is the frame of the press
  texelFetch(ch, ivec2(code, 0), 0).x   key held
  texelFetch(ch, ivec2(code, 1), 0).x   key pressed this frame
  texelFetch(ch, ivec2(code, 2), 0).x   key toggle
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _parse_resolution(text: str) -> tuple[int, int]:
    cleaned = str(text).lower().replace("×", "x").replace(",", "x")
    parts = [p for p in cleaned.split("x") if p]
    if len(parts) != 2:
        raise ValueError(f"resolution must look like 640x360 (got {text!r})")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"resolution must be integers (got {text!r})") from None
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive (got {text!r})")
    return width, height


def _build_settings(args: argparse.Namespace, project: Any) -> Any:
    from .inputs import InputTimeline, load_input_spec
    from .renderer import (
        DEFAULT_DATE,
        DEFAULT_MAX_FRAMES,
        PRECHARGE_ALL,
        RenderSettings,
    )

    width = project.default("width", 640)
    height = project.default("height", 360)
    if getattr(args, "resolution", None):
        width, height = _parse_resolution(args.resolution)
    # `is not None` throughout, not truthiness: 0 is a value the user typed and
    # must be rejected, not silently replaced by the default.
    if getattr(args, "width", None) is not None:
        width = args.width
    if getattr(args, "height", None) is not None:
        height = args.height
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive (got {width}x{height})")

    fps = (
        args.fps
        if getattr(args, "fps", None) is not None
        else float(project.default("fps", 60.0))
    )
    if fps <= 0:
        raise ValueError(f"--fps must be positive (got {fps})")

    frame = args.frame if getattr(args, "frame", None) is not None else 0
    if frame < 0:
        raise ValueError(f"--frame must be >= 0 (got {frame})")

    raw_precharge = getattr(args, "precharge", None)
    precharge: int | str | None = None
    if raw_precharge is not None:
        text = str(raw_precharge).strip().lower()
        if text in ("all", "full"):
            precharge = PRECHARGE_ALL
        else:
            try:
                precharge = int(text)
            except ValueError:
                raise ValueError(
                    f"--precharge expects a frame count or 'all' (got {raw_precharge!r})"
                ) from None
            if precharge < 0:
                raise ValueError(f"--precharge must be >= 0 (got {precharge})")

    # Parsed after fps, since "time" in an operation is converted using it.
    spec = getattr(args, "input_spec", None)
    timeline = load_input_spec(spec, fps) if spec else InputTimeline.empty(fps)

    date = DEFAULT_DATE
    if getattr(args, "date", None):
        parts = [p.strip() for p in str(args.date).split(",")]
        if len(parts) != 4:
            raise ValueError("--date expects four numbers: Y,M,D,S")
        date = tuple(float(p) for p in parts)  # type: ignore[assignment]

    return RenderSettings(
        width=int(width),
        height=int(height),
        fps=float(fps),
        frame=int(frame),
        inputs=timeline,
        date=date,
        precharge=precharge,
        max_frames=(
            args.max_frames
            if getattr(args, "max_frames", None) is not None
            else DEFAULT_MAX_FRAMES
        ),
    )


def _open_context(args: argparse.Namespace) -> Any:
    from .context import create_context

    return create_context(
        device_index=getattr(args, "device", None),
        require=getattr(args, "gl_version", None),
        backend=getattr(args, "backend", None),
        allow_software=bool(getattr(args, "allow_software", False)),
    )


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    """Flags choosing which frames get saved or compared.

    Frames are always an arithmetic progression: start at --frame, take --count
    of them, spaced --every apart. A single invocation therefore cannot capture an
    arbitrary set like 0,7,53 -- run it twice for that -- but every regular
    sampling pattern is expressible without a bespoke range syntax.
    """
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="capture N frames instead of one, starting at --frame",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=None,
        metavar="M",
        help="frame separation between captures, for use with --count (default: 1)",
    )


def _resolve_capture(args: argparse.Namespace, start: int) -> list[int]:
    """Frames to capture: ``start``, then ``count`` of them spaced ``every``."""
    count = getattr(args, "count", None)
    every = getattr(args, "every", None)

    if count is None:
        if every is not None:
            raise ValueError("--every only applies together with --count")
        return [start]

    if count < 1:
        raise ValueError(f"--count must be >= 1 (got {count})")
    step = 1 if every is None else every
    if step < 1:
        raise ValueError(f"--every must be >= 1 (got {step})")
    return [start + index * step for index in range(count)]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """Report GPU/EGL environment and, if present, the project layout."""
    from .context import ContextError, enumerate_devices, probe_devices
    from .project import ProjectError, load_project
    from .selftest import check_all_devices

    report = _Reporter(args.json, args.quiet)
    payload: dict[str, Any] = {"version": __version__}

    report.say(f"shadertoy-local {__version__}")
    report.say()

    runtime_ok: list[bool] = []
    if getattr(args, "runtime_check", True):
        # Actually compile and run a shader on every device: enumerating one is
        # no promise it can bind, and binding is no promise it renders correctly.
        devices = enumerate_devices()
        checks = check_all_devices(require=getattr(args, "gl_version", None))
        by_index = {c.device_index: c for c in checks}
        payload["devices"] = [
            {
                **dev.to_dict(),
                # The check reports the renderer it actually bound, which is more
                # trustworthy than a pre-context guess.
                "renderer": (by_index.get(dev.index).renderer
                             if dev.index in by_index else dev.renderer),
                "runtime": by_index[dev.index].to_dict()
                if dev.index in by_index
                else None,
            }
            for dev in devices
        ]
        if not devices:
            payload["devices"] = []
            payload["runtime"] = [c.to_dict() for c in checks]

        if devices:
            report.say("EGL devices (with shader runtime check):")
            for dev in devices:
                check = by_index.get(dev.index)
                kind = "software" if dev.is_software else "hardware"
                label = (check.renderer if check and check.renderer else dev.label)
                report.say(f"  [{dev.index}] {label}  ({kind})")
                if check is not None:
                    runtime_ok.append(check.ok)
                    line = f"        runtime: {check.summary()}"
                    if check.ok:
                        report.say(line)
                    else:
                        report.warn(line)
        else:
            report.say("EGL devices: none enumerated")
            for check in checks:
                runtime_ok.append(check.ok)
                line = f"  default device runtime: {check.summary()}"
                report.say(line) if check.ok else report.warn(line)
    else:
        devices = probe_devices()
        payload["devices"] = [d.to_dict() for d in devices]
        if devices:
            report.say("EGL devices:")
            for dev in devices:
                kind = "software" if dev.is_software else "hardware"
                report.say(f"  [{dev.index}] {dev.label}  ({kind})")
        else:
            report.say("EGL devices: none found")

    try:
        handle = _open_context(args)
    except ContextError as exc:
        payload["context"] = None
        payload["error"] = str(exc)
        report.warn(f"\nerror: {exc}")
        report.emit(payload)
        return EXIT_ENVIRONMENT

    payload["context"] = handle.to_dict()
    report.say()
    report.say("Active context:")
    for key, value in handle.to_dict().items():
        report.say(f"  {key:<16} {value}")
    handle.release()

    try:
        project = load_project(args.directory)
    except ProjectError as exc:
        payload["project"] = None
        payload["project_error"] = str(exc)
    else:
        from .wiring import build_report, format_report

        wiring = build_report(project)
        payload["project"] = project.to_dict()
        payload["porting"] = wiring.to_dict()

        report.say()
        title = f"Project: {project.root}"
        name = project.config.get("name")
        if isinstance(name, str) and name:
            title += f"  ({name})"
        report.say(title)
        report.say()
        for entry in project.ordered_files:
            detail = ""
            if entry.spec is not None:
                channels = ", ".join(
                    f"iChannel{i}={b.source}"
                    for i, b in sorted(entry.spec.channels.items())
                )
                if entry.spec.scale != 1.0:
                    detail += f" scale={entry.spec.scale}"
                if channels:
                    detail += f"  [{channels}]"
            report.say(f"  {entry.label:<9} {entry.path.name}{detail}")

        if not getattr(args, "porting", True):
            report.emit(payload)
            return EXIT_OK
        report.say()
        report.say("-- porting to shadertoy.com " + "-" * 44)
        for line in format_report(wiring):
            report.say(line)

    if runtime_ok and not any(runtime_ok):
        report.warn(
            "\nerror: no device could compile and run a shader; "
            "rendering will not work here"
        )
        payload["ok"] = False
        report.emit(payload)
        return EXIT_ENVIRONMENT

    payload["ok"] = True
    report.emit(payload)
    return EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold a new project."""
    from .templates import DEFAULT_TEMPLATE, TEMPLATES, scaffold

    report = _Reporter(args.json, args.quiet)
    if args.list:
        report.say("Available templates:")
        for name in sorted(TEMPLATES):
            files = ", ".join(sorted(TEMPLATES[name]))
            report.say(f"  {name:<10} {files}")
        report.emit({"templates": sorted(TEMPLATES)})
        return EXIT_OK

    root = Path(args.directory).expanduser().resolve()
    try:
        written = scaffold(root, args.template or DEFAULT_TEMPLATE, force=args.force)
    except (ValueError, FileExistsError) as exc:
        report.warn(f"error: {exc}")
        report.emit({"error": str(exc)})
        return EXIT_USAGE

    report.say(f"Created {args.template or DEFAULT_TEMPLATE} project in {root}")
    for path in written:
        report.say(f"  {path.relative_to(root)}")
    report.say()
    report.say("Next: shadertoy render")
    report.emit(
        {
            "root": str(root),
            "template": args.template or DEFAULT_TEMPLATE,
            "files": [str(p) for p in written],
        }
    )
    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    """Compile every pass without rendering."""
    from .context import ContextError
    from .diagnostics import format_diagnostics, summarize
    from .portability import EXPLANATIONS, lint_all
    from .project import ProjectError, load_project
    from .renderer import Renderer, RenderSettings

    report = _Reporter(args.json, args.quiet)
    try:
        project = load_project(args.directory)
    except ProjectError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_USAGE

    try:
        handle = _open_context(args)
    except ContextError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_ENVIRONMENT

    renderer = Renderer(project, handle.ctx, RenderSettings())
    try:
        try:
            diagnostics = renderer.compile(collect_all=True)
        except ProjectError as exc:
            report.warn(f"error: {exc}")
            report.emit({"ok": False, "error": str(exc)})
            return EXIT_USAGE

        errors, warnings = summarize(diagnostics)
        composed = {name: rp.composed for name, rp in renderer.passes.items()}
        by_pass: dict[str, list] = {}
        for diag in diagnostics:
            by_pass.setdefault(diag.pass_name or "?", []).append(diag)

        for pass_name, group in by_pass.items():
            text = format_diagnostics(
                group, composed.get(pass_name), color=sys.stderr.isatty()
            )
            if text:
                report.warn(text)

        # Warn where this project would behave differently on shadertoy.com.
        # On by default: warnings only, and each predicts a real divergence that
        # compiles cleanly here.
        portability: list = []
        if getattr(args, "portability", True):
            portability = lint_all(project)
            for diag in portability:
                report.warn(
                    f"{diag.location()}: {diag.severity} [{diag.code}]: "
                    f"{diag.message}"
                )
            # One footer per distinct check that fired, not per occurrence.
            for code in dict.fromkeys(d.code for d in portability):
                note = EXPLANATIONS.get(code)
                if note:
                    report.warn(f"note [{code}]: {note}")
            # Severity reflects consequence: a Common-tab finding is cosmetic on
            # the site, a struct ternary does not compile there at all, so it
            # counts as an error and fails the command.
            errors += sum(1 for d in portability if d.is_error)
            warnings += sum(1 for d in portability if not d.is_error)

        ok = errors == 0
        if ok:
            names = ", ".join(s.label for s in project.ordered_passes)
            suffix = f", {len(portability)} portability warning(s)" if portability else ""
            report.say(
                f"ok: {len(project.ordered_passes)} pass(es) compiled "
                f"[{names}]{suffix}"
            )
        else:
            report.warn(f"failed: {errors} error(s), {warnings} warning(s)")

        report.emit(
            {
                "ok": ok,
                "errors": errors,
                "warnings": warnings,
                "passes": [s.name for s in project.ordered_passes],
                "diagnostics": [d.to_dict() for d in diagnostics],
                "portability": [d.to_dict() for d in portability],
            }
        )
        return EXIT_OK if ok else EXIT_FAILED
    finally:
        renderer.release()
        handle.release()


def _prepare_render(args: argparse.Namespace, report: _Reporter):
    """Load project, open a context, compile. Returns (project, handle, renderer)
    or an exit code on failure."""
    from .context import ContextError
    from .diagnostics import format_diagnostics
    from .project import ProjectError, load_project
    from .renderer import Renderer, ShaderCompileError

    try:
        project = load_project(args.directory)
    except ProjectError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_USAGE

    try:
        settings = _build_settings(args, project)
    except ValueError as exc:
        # InputError subclasses ValueError, so bad --mouse/--key land here too.
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_USAGE

    try:
        handle = _open_context(args)
    except ContextError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_ENVIRONMENT

    renderer = Renderer(project, handle.ctx, settings)
    try:
        renderer.compile()
    except ShaderCompileError as exc:
        report.warn(
            format_diagnostics(
                exc.diagnostics, exc.composed, color=sys.stderr.isatty()
            )
        )
        report.warn("failed: shader did not compile")
        report.emit(
            {
                "ok": False,
                "error": "compile failed",
                "diagnostics": [d.to_dict() for d in exc.diagnostics],
            }
        )
        renderer.release()
        handle.release()
        return EXIT_FAILED
    except ProjectError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        renderer.release()
        handle.release()
        return EXIT_USAGE

    return project, handle, renderer, settings


def cmd_render(args: argparse.Namespace) -> int:
    """Render one or more frames to PNG."""
    from .analysis import frame_stats, parse_probe, run_probe, save_png
    from .golden import capture_name

    report = _Reporter(args.json, args.quiet)
    prepared = _prepare_render(args, report)
    if isinstance(prepared, int):
        return prepared
    project, handle, renderer, settings = prepared

    try:
        try:
            frames = _resolve_capture(args, settings.frame)
        except ValueError as exc:
            report.warn(f"error: {exc}")
            report.emit({"ok": False, "error": str(exc)})
            return EXIT_USAGE

        try:
            probes = [parse_probe(p) for p in (args.probe or [])]
        except ValueError as exc:
            report.warn(f"error: {exc}")
            report.emit({"ok": False, "error": str(exc)})
            return EXIT_USAGE

        wanted_passes = _resolve_passes(args.pass_name, renderer, report)
        if wanted_passes is None:
            return EXIT_USAGE

        outdir = Path(args.output or project.root / "out").expanduser()
        results: list[dict[str, Any]] = []
        failures = 0

        for capture in renderer.run(frames):
            entry: dict[str, Any] = {
                "frame": capture.frame,
                "time": capture.time,
                "duration_ms": round(capture.duration_ms, 4),
                "passes": {},
            }
            for name in wanted_passes:
                array = capture.images.get(name)
                if array is None:
                    continue
                item: dict[str, Any] = {}
                if not args.no_write:
                    filename = (
                        f"{capture_name(name, capture.frame)}.png"
                        if len(frames) > 1 or name != "image"
                        else "image.png"
                    )
                    path = save_png(
                        array, outdir / filename, opaque=not args.keep_alpha
                    )
                    item["file"] = str(path)
                if args.stats:
                    item["stats"] = frame_stats(array)
                if probes and name == "image":
                    item["probes"] = [run_probe(array, p) for p in probes]
                    failures += sum(
                        1
                        for r in item["probes"]
                        if r.get("passed") is False
                    )
                entry["passes"][name] = item
            results.append(entry)

        total_ms = sum(r["duration_ms"] for r in results)
        report.say(
            f"rendered {len(results)} frame(s) at {settings.width}x{settings.height} "
            f"on {handle.ctx.info['GL_RENDERER']}"
        )
        for entry in results:
            for name, item in entry["passes"].items():
                if "file" in item:
                    report.say(f"  {item['file']}")
        if args.stats:
            for entry in results:
                stats = entry["passes"].get("image", {}).get("stats")
                if stats:
                    report.say(
                        f"  frame {entry['frame']}: "
                        f"luma mean {stats['luma']['mean']:.4f}, "
                        f"{stats['unique_colors']} colours, "
                        f"finite={stats['finite']}"
                    )
        for entry in results:
            for item in entry["passes"].values():
                for probe in item.get("probes", []):
                    if "passed" in probe:
                        mark = "ok" if probe["passed"] else "FAIL"
                        line = (
                            f"  probe ({probe['x']},{probe['y']}) {mark}: "
                            f"got {[round(v,4) for v in probe['rgba']]} "
                            f"expected {probe['expected']} "
                            f"(max diff {probe['max_diff']:.5f})"
                        )
                        if probe["passed"]:
                            report.say(line)
                        else:
                            report.warn(line)
        report.say(f"  total gpu time {total_ms:.3f} ms")

        ok = failures == 0
        report.emit(
            {
                "ok": ok,
                "settings": settings.to_dict(),
                "renderer": handle.to_dict(),
                "output_dir": str(outdir),
                "frames": results,
                "probe_failures": failures,
            }
        )
        return EXIT_OK if ok else EXIT_FAILED
    finally:
        renderer.release()
        handle.release()


def _resolve_passes(
    requested: list[str] | None, renderer: Any, report: _Reporter
) -> list[str] | None:
    """Which passes to output. Defaults to the image pass alone."""
    available = [s.name for s in renderer.project.ordered_passes]
    if not requested:
        return ["image"]
    if "all" in requested:
        return available
    unknown = [r for r in requested if r not in available]
    if unknown:
        message = (
            f"unknown pass(es): {', '.join(unknown)}. "
            f"Available: {', '.join(available)}, or 'all'"
        )
        report.warn(f"error: {message}")
        report.emit({"ok": False, "error": message})
        return None
    return [name for name in available if name in requested]


def cmd_probe(args: argparse.Namespace) -> int:
    """Query pixel values, optionally asserting expected colours."""
    from .analysis import AnalysisError, parse_probe, run_probe

    report = _Reporter(args.json, args.quiet)
    if not args.probe:
        report.warn("error: probe requires at least one --at X,Y")
        report.emit({"ok": False, "error": "no probes given"})
        return EXIT_USAGE

    try:
        probes = [parse_probe(p) for p in args.probe]
    except AnalysisError as exc:
        report.warn(f"error: {exc}")
        report.emit({"ok": False, "error": str(exc)})
        return EXIT_USAGE
    for probe in probes:
        probe.tolerance = args.tolerance

    prepared = _prepare_render(args, report)
    if isinstance(prepared, int):
        return prepared
    project, handle, renderer, settings = prepared

    try:
        wanted = _resolve_passes(args.pass_name, renderer, report)
        if wanted is None:
            return EXIT_USAGE

        capture = renderer.render_frame()
        payload: dict[str, Any] = {
            "ok": True,
            "frame": capture.frame,
            "time": capture.time,
            "passes": {},
        }
        failures = 0
        for name in wanted:
            array = capture.images.get(name)
            if array is None:
                continue
            results = [run_probe(array, p) for p in probes]
            failures += sum(1 for r in results if r.get("passed") is False)
            payload["passes"][name] = results
            for result in results:
                rgba = [round(v, 6) for v in result["rgba"]]
                line = f"{name} ({result['x']},{result['y']}) = {rgba}"
                if "passed" in result:
                    line += (
                        f"  expected {result['expected']}  "
                        f"maxdiff {result['max_diff']:.6f}  "
                        f"{'ok' if result['passed'] else 'FAIL'}"
                    )
                if not result["finite"]:
                    line += "  [NON-FINITE]"
                # A failed assertion is reportable even under --json.
                if result.get("passed") is False:
                    report.warn(line)
                else:
                    report.say(line)

        payload["ok"] = failures == 0
        payload["failures"] = failures
        report.emit(payload)
        return EXIT_OK if failures == 0 else EXIT_FAILED
    finally:
        renderer.release()
        handle.release()


def cmd_stats(args: argparse.Namespace) -> int:
    """Report frame statistics, with optional assertions."""
    from .analysis import frame_stats, histogram

    report = _Reporter(args.json, args.quiet)
    prepared = _prepare_render(args, report)
    if isinstance(prepared, int):
        return prepared
    project, handle, renderer, settings = prepared

    try:
        wanted = _resolve_passes(args.pass_name, renderer, report)
        if wanted is None:
            return EXIT_USAGE

        capture = renderer.render_frame()
        payload: dict[str, Any] = {
            "ok": True,
            "frame": capture.frame,
            "time": capture.time,
            "passes": {},
        }
        problems: list[str] = []

        for name in wanted:
            array = capture.images.get(name)
            if array is None:
                continue
            stats = frame_stats(array)
            if args.histogram:
                stats["histogram"] = histogram(array, bins=args.bins)
            payload["passes"][name] = stats

            report.say(f"{name}: {stats['width']}x{stats['height']}")
            for channel in ("r", "g", "b", "a"):
                info = stats["channels"][channel]
                report.say(
                    f"  {channel}: min {_fmt(info['min'])}  "
                    f"max {_fmt(info['max'])}  mean {_fmt(info['mean'])}"
                )
            report.say(
                f"  luma mean {_fmt(stats['luma']['mean'])}  "
                f"unique colours {stats['unique_colors']}  "
                f"clipped {stats['fraction_clipped'] * 100:.2f}%"
            )
            if stats["has_nan"] or stats["has_inf"]:
                report.warn(
                    f"  {name}: non-finite values present -- "
                    f"{stats['nan_count']} NaN, {stats['inf_count']} Inf"
                )

            # Assertions: these are what let a CI job fail on a broken shader.
            if args.assert_finite and not stats["finite"]:
                problems.append(
                    f"{name}: {stats['nan_count']} NaN and {stats['inf_count']} Inf "
                    "values present"
                )
            if args.assert_not_black and stats["is_black"]:
                problems.append(f"{name}: frame is entirely black")
            if args.assert_not_uniform and stats["is_uniform"]:
                problems.append(f"{name}: frame is a single uniform colour")
            if args.min_unique_colors and stats["unique_colors"] < args.min_unique_colors:
                problems.append(
                    f"{name}: only {stats['unique_colors']} unique colours, "
                    f"expected at least {args.min_unique_colors}"
                )

        for problem in problems:
            report.warn(f"FAIL {problem}")
        payload["ok"] = not problems
        payload["problems"] = problems
        report.emit(payload)
        return EXIT_OK if not problems else EXIT_FAILED
    finally:
        renderer.release()
        handle.release()


def _golden_run(args: argparse.Namespace, bless: bool) -> int:
    """Shared implementation of `test` and `bless`."""
    from .golden import capture_name, compare, write_golden

    report = _Reporter(args.json, args.quiet)
    prepared = _prepare_render(args, report)
    if isinstance(prepared, int):
        return prepared
    project, handle, renderer, settings = prepared

    try:
        try:
            frames = _resolve_capture(args, settings.frame)
        except ValueError as exc:
            report.warn(f"error: {exc}")
            report.emit({"ok": False, "error": str(exc)})
            return EXIT_USAGE

        wanted = _resolve_passes(args.pass_name, renderer, report)
        if wanted is None:
            return EXIT_USAGE

        artifacts = (
            Path(args.artifacts).expanduser()
            if getattr(args, "artifacts", None)
            else project.root / "out" / "failures"
        )
        entries: list[dict[str, Any]] = []
        failures = 0
        blessed: list[str] = []

        for capture in renderer.run(frames):
            for name in wanted:
                array = capture.images.get(name)
                if array is None:
                    continue
                key = capture_name(name, capture.frame)
                if bless:
                    path = write_golden(array, project.root, key)
                    blessed.append(str(path))
                    entries.append(
                        {"name": key, "status": "written", "golden": str(path)}
                    )
                    report.say(f"  wrote {path.relative_to(project.root)}")
                    continue
                result = compare(
                    array,
                    project.root,
                    key,
                    max_diff=args.max_diff,
                    mean_diff=args.mean_diff,
                    write_artifacts=artifacts,
                )
                entries.append(result.to_dict())
                if not result.passed:
                    failures += 1
                    report.warn(f"FAIL {key}: {result.message}")
                    if result.diff_path:
                        report.warn(f"     diff: {result.diff_path}")
                else:
                    report.say(
                        f"  ok {key}  max diff {result.max_diff}, "
                        f"mean {result.mean_diff:.3f}"
                    )

        if bless:
            report.say(f"blessed {len(blessed)} reference image(s)")
            report.emit({"ok": True, "blessed": blessed, "results": entries})
            return EXIT_OK

        total = len(entries)
        if failures:
            report.warn(f"failed: {failures}/{total} comparison(s) differ")
        else:
            report.say(f"ok: {total}/{total} comparison(s) match")
        report.emit(
            {
                "ok": failures == 0,
                "total": total,
                "failures": failures,
                "results": entries,
            }
        )
        return EXIT_OK if failures == 0 else EXIT_FAILED
    finally:
        renderer.release()
        handle.release()


def cmd_test(args: argparse.Namespace) -> int:
    """Compare renders against committed reference images."""
    return _golden_run(args, bless=False)


def cmd_bless(args: argparse.Namespace) -> int:
    """Write or update reference images."""
    return _golden_run(args, bless=True)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadertoy",
        description=(
            "Headless local Shadertoy runtime and shader test harness. "
            "Treats a directory as a project: image.glsl plus optional "
            "common.glsl and buffer_a..d.glsl, wired up by shadertoy.json."
        ),
        epilog=(
            "Exit codes: 0 ok, 1 shader or assertion failure, 2 usage/project "
            "error, 3 no usable GPU."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # info
    info = subparsers.add_parser(
        "info", help="show GPU, EGL devices, and project layout"
    )
    _add_project_args(info)
    _add_gpu_args(info)
    info.add_argument(
        "--no-porting",
        dest="porting",
        action="store_false",
        default=True,
        help="omit the shadertoy.com porting guide from the project section",
    )
    info.add_argument(
        "--no-runtime-check",
        dest="runtime_check",
        action="store_false",
        default=True,
        help=(
            "only enumerate devices; skip compiling and running a verification "
            "shader on each one"
        ),
    )
    info.set_defaults(func=cmd_info)

    # init
    init = subparsers.add_parser("init", help="scaffold a new project")
    _add_project_args(init)
    init.add_argument(
        "-t", "--template", default=None, help="template name (default: basic)"
    )
    init.add_argument("--list", action="store_true", help="list available templates")
    init.add_argument("--force", action="store_true", help="overwrite existing files")
    init.set_defaults(func=cmd_init)

    # check
    check = subparsers.add_parser(
        "check", help="compile all passes and report errors (no rendering)"
    )
    _add_project_args(check)
    _add_gpu_args(check)
    check.add_argument(
        "--no-portability",
        dest="portability",
        action="store_false",
        default=True,
        help=(
            "skip shadertoy.com portability checks: Common-tab uniform "
            "visibility, and ?: on struct types (both compile here but diverge "
            "on the site)"
        ),
    )
    check.set_defaults(func=cmd_check)

    # render
    render = subparsers.add_parser("render", help="render frames to PNG")
    _add_project_args(render)
    _add_gpu_args(render)
    _add_frame_args(render)
    _add_input_args(render)
    render.add_argument(
        "-o", "--output", default=None, metavar="DIR", help="output directory"
    )
    _add_capture_args(render)
    render.add_argument(
        "-p",
        "--pass",
        dest="pass_name",
        action="append",
        default=None,
        metavar="NAME",
        help="pass to write: image, buffer_a, ..., or all (default: image)",
    )
    render.add_argument(
        "--stats", action="store_true", help="include frame statistics"
    )
    render.add_argument(
        "--probe",
        action="append",
        default=None,
        metavar="SPEC",
        help="probe a pixel: X,Y or X,Y=R,G,B[,A]",
    )
    render.add_argument(
        "--keep-alpha",
        action="store_true",
        help="preserve alpha instead of forcing it opaque",
    )
    render.add_argument(
        "--no-write", action="store_true", help="render but do not write files"
    )
    render.set_defaults(func=cmd_render)

    # probe
    probe = subparsers.add_parser(
        "probe", help="read pixel values, optionally asserting them"
    )
    _add_project_args(probe)
    _add_gpu_args(probe)
    _add_frame_args(probe)
    _add_input_args(probe)
    probe.add_argument(
        "--at",
        dest="probe",
        action="append",
        default=None,
        metavar="SPEC",
        help="pixel to read: X,Y or X,Y=R,G,B[,A]; repeatable",
    )
    probe.add_argument(
        "--tolerance",
        type=float,
        default=1 / 255,
        help="tolerance for expected values (default: 1/255)",
    )
    probe.add_argument(
        "-p",
        "--pass",
        dest="pass_name",
        action="append",
        default=None,
        metavar="NAME",
        help="pass to probe (default: image)",
    )
    probe.set_defaults(func=cmd_probe)

    # stats
    stats = subparsers.add_parser("stats", help="report frame statistics")
    _add_project_args(stats)
    _add_gpu_args(stats)
    _add_frame_args(stats)
    _add_input_args(stats)
    stats.add_argument(
        "-p",
        "--pass",
        dest="pass_name",
        action="append",
        default=None,
        metavar="NAME",
        help="pass to analyse (default: image)",
    )
    stats.add_argument("--histogram", action="store_true", help="include a histogram")
    stats.add_argument("--bins", type=int, default=16, help="histogram bins")
    stats.add_argument(
        "--assert-finite",
        action="store_true",
        help="fail if any NaN or Inf is present",
    )
    stats.add_argument(
        "--assert-not-black", action="store_true", help="fail if the frame is all black"
    )
    stats.add_argument(
        "--assert-not-uniform",
        action="store_true",
        help="fail if the frame is a single flat colour",
    )
    stats.add_argument(
        "--min-unique-colors",
        type=int,
        default=None,
        metavar="N",
        help="fail if fewer than N distinct colours",
    )
    stats.set_defaults(func=cmd_stats)

    # test / bless
    for name, help_text, func in (
        ("test", "compare renders against golden images", cmd_test),
        ("bless", "write or update golden images", cmd_bless),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_project_args(sub)
        _add_gpu_args(sub)
        _add_frame_args(sub)
        _add_input_args(sub)
        _add_capture_args(sub)
        sub.add_argument(
            "-p",
            "--pass",
            dest="pass_name",
            action="append",
            default=None,
            metavar="NAME",
            help="pass to compare (default: image)",
        )
        if name == "test":
            sub.add_argument(
                "--max-diff",
                type=int,
                default=2,
                help="max allowed per-channel difference in 8-bit levels (default: 2)",
            )
            sub.add_argument(
                "--mean-diff",
                type=float,
                default=0.5,
                help="max allowed mean per-channel difference (default: 0.5)",
            )
            sub.add_argument(
                "--artifacts",
                default=None,
                metavar="DIR",
                help="where to write actual/diff images on failure",
            )
        sub.set_defaults(func=func)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "help_input", False):
        print(INPUT_HELP, file=sys.stderr)
        return EXIT_OK
    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    from .inputs import InputError
    from .project import ProjectError
    from .renderer import RenderError

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (InputError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "json", False):
            json.dump({"ok": False, "error": str(exc)}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        return EXIT_USAGE
    except (ProjectError, RenderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "json", False):
            json.dump({"ok": False, "error": str(exc)}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
