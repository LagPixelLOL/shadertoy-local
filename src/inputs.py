"""Simulated input as a single timeline of operations.

Shadertoy shaders are interactive, so a harness that cannot press a key or drag
the mouse can only exercise a fraction of one. Earlier versions of this module
took a static mouse position and a static set of held keys, which meant every
simulated frame saw *identical* input -- a drag, a click-and-release, or a key
pressed at frame 20 could not be expressed at all.

Input is therefore one ordered list of operations, covering pointer and keyboard
together, each scheduled at a frame or a time:

.. code-block:: json

    [
      {"frame": 0,   "op": "mouse_down", "pos": [320, 180]},
      {"frame": 20,  "op": "mouse_move", "pos": [400, 220]},
      {"time":  0.5, "op": "mouse_up"},
      {"frame": 5,   "op": "key_down",   "keys": ["w", "shift"]},
      {"time":  1.0, "op": "key_up",     "keys": ["w"]},
      {"frame": 40,  "op": "key_tap",    "keys": ["space"]},
      {"frame": 45,  "op": "key_toggle", "keys": ["g"]}
    ]

Everything is derived from the frame index, so a run remains reproducible.

The two devices are modelled exactly as Shadertoy exposes them.

``iMouse``
    ``xy`` is the cursor in pixels. ``zw`` encodes where the press began, with
    button state folded into the signs: ``z > 0`` while held, ``w > 0`` only on
    the frame of the press, both negative once released.

Keyboard channel
    A 256x3 texture indexed by JavaScript key code, read with ``texelFetch``:
    row 0 is held, row 1 is pressed-this-frame, row 2 is a toggle.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Width of the keyboard texture (one column per JavaScript key code).
KEYBOARD_WIDTH = 256
#: Rows: 0 = held, 1 = pressed this frame, 2 = toggle.
KEYBOARD_HEIGHT = 3

#: Human-writable names mapped to JavaScript ``keyCode`` values, which is what
#: Shadertoy indexes the keyboard texture by.
KEY_CODES: dict[str, int] = {
    "backspace": 8, "tab": 9, "enter": 13, "return": 13, "shift": 16,
    "ctrl": 17, "control": 17, "alt": 18, "capslock": 20, "escape": 27,
    "esc": 27, "space": 32, "pageup": 33, "pagedown": 34, "end": 35,
    "home": 36, "left": 37, "up": 38, "right": 39, "down": 40,
    "insert": 45, "delete": 46,
    **{str(d): 48 + d for d in range(10)},
    **{chr(ord("a") + i): 65 + i for i in range(26)},
    **{f"numpad{d}": 96 + d for d in range(10)},
    **{f"f{n}": 111 + n for n in range(1, 13)},
    "semicolon": 186, "equal": 187, "comma": 188, "minus": 189,
    "period": 190, "slash": 191, "backquote": 192,
    "bracketleft": 219, "backslash": 220, "bracketright": 221, "quote": 222,
}

#: Operations that move or click the pointer.
MOUSE_OPS = ("mouse_down", "mouse_up", "mouse_move")
#: Operations that act on keys.
KEY_OPS = ("key_down", "key_up", "key_tap", "key_toggle", "key_untoggle")
OPS = (*MOUSE_OPS, *KEY_OPS)


class InputError(ValueError):
    """Raised for malformed input specifications."""


# --------------------------------------------------------------------------
# Key parsing
# --------------------------------------------------------------------------


def parse_key(token: Any) -> int:
    """Resolve a key name or numeric code to a JavaScript key code."""
    if isinstance(token, bool):
        raise InputError(f"key must be a name or code, got {token!r}")
    if isinstance(token, int):
        code = token
        if not 0 <= code < KEYBOARD_WIDTH:
            raise InputError(f"key code {code} out of range 0..{KEYBOARD_WIDTH - 1}")
        return code
    text = str(token).strip()
    if not text:
        raise InputError("empty key specification")
    lowered = text.lower()
    if lowered in KEY_CODES:
        return KEY_CODES[lowered]
    if text.isdigit():
        code = int(text)
        if not 0 <= code < KEYBOARD_WIDTH:
            raise InputError(f"key code {code} out of range 0..{KEYBOARD_WIDTH - 1}")
        return code
    raise InputError(
        f"unknown key {text!r}. Use a name (a-z, 0-9, left, space, f1, ...) "
        f"or a numeric JavaScript key code."
    )


def parse_keys(tokens: Iterable[Any]) -> list[int]:
    """Parse a list, or a comma-separated string, of key specs."""
    if isinstance(tokens, (str, int)):
        tokens = [tokens]
    codes: list[int] = []
    for token in tokens:
        if isinstance(token, str):
            for part in token.split(","):
                part = part.strip()
                if part:
                    codes.append(parse_key(part))
        else:
            codes.append(parse_key(token))
    return codes


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InputEvent:
    """One scheduled operation."""

    frame: int
    op: str
    pos: tuple[float, float] | None = None
    keys: tuple[int, ...] = ()
    normalized: bool = False
    #: Original ``time`` value, kept only so reports can echo what was written.
    time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"frame": self.frame, "op": self.op}
        if self.time is not None:
            out["time"] = self.time
        if self.pos is not None:
            out["pos"] = list(self.pos)
            if self.normalized:
                out["normalized"] = True
        if self.keys:
            out["keys"] = list(self.keys)
        return out


def _parse_pos(raw: Any, where: str) -> tuple[float, float]:
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise InputError(f"{where}: pos must be [x, y] or \"x,y\"")
    if len(parts) != 2:
        raise InputError(f"{where}: pos needs exactly two numbers, got {raw!r}")
    try:
        return float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        raise InputError(f"{where}: pos values must be numbers, got {raw!r}") from None


def _parse_when(spec: dict[str, Any], fps: float, where: str) -> tuple[int, float | None]:
    """Resolve ``frame`` or ``time`` into a frame index.

    Both spellings are accepted because both are natural: frames when reasoning
    about simulation steps, seconds when matching what a shader does with
    ``iTime``.
    """
    has_frame = "frame" in spec
    has_time = "time" in spec
    if has_frame and has_time:
        raise InputError(f"{where}: give either \"frame\" or \"time\", not both")
    if not has_frame and not has_time:
        raise InputError(f"{where}: needs a \"frame\" or \"time\"")

    if has_frame:
        frame = spec["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise InputError(f"{where}: \"frame\" must be an integer, got {frame!r}")
        if frame < 0:
            raise InputError(f"{where}: \"frame\" must be >= 0, got {frame}")
        return frame, None

    seconds = spec["time"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise InputError(f"{where}: \"time\" must be a number, got {seconds!r}")
    if seconds < 0:
        raise InputError(f"{where}: \"time\" must be >= 0, got {seconds}")
    if fps <= 0:
        raise InputError(f"{where}: cannot convert \"time\" with fps={fps}")
    # Round to nearest so 0.5s at 60fps is frame 30, not 29.
    return int(round(float(seconds) * fps)), float(seconds)


def parse_event(spec: Any, fps: float, index: int) -> InputEvent:
    """Validate and normalise one operation."""
    where = f"input[{index}]"
    if not isinstance(spec, dict):
        raise InputError(f"{where}: each operation must be an object, got {type(spec).__name__}")

    unknown = sorted(set(spec) - {"frame", "time", "op", "pos", "keys", "normalized"})
    if unknown:
        raise InputError(
            f"{where}: unknown key(s): {', '.join(unknown)}. "
            f"Expected any of: frame, time, op, pos, keys, normalized"
        )

    op = spec.get("op")
    if not isinstance(op, str) or op not in OPS:
        raise InputError(
            f"{where}: \"op\" must be one of {', '.join(OPS)}, got {op!r}"
        )

    frame, seconds = _parse_when(spec, fps, where)

    pos = None
    if "pos" in spec:
        if op not in MOUSE_OPS:
            raise InputError(f"{where}: \"pos\" does not apply to {op}")
        pos = _parse_pos(spec["pos"], where)
    elif op == "mouse_move":
        raise InputError(f"{where}: mouse_move needs a \"pos\"")

    keys: tuple[int, ...] = ()
    if "keys" in spec:
        if op not in KEY_OPS:
            raise InputError(f"{where}: \"keys\" does not apply to {op}")
        keys = tuple(dict.fromkeys(parse_keys(spec["keys"])))
        if not keys:
            raise InputError(f"{where}: \"keys\" is empty")
    elif op in KEY_OPS:
        raise InputError(f"{where}: {op} needs \"keys\"")

    normalized = bool(spec.get("normalized", False))
    if normalized and pos is None:
        raise InputError(f"{where}: \"normalized\" only applies alongside \"pos\"")

    return InputEvent(
        frame=frame, op=op, pos=pos, keys=keys, normalized=normalized, time=seconds
    )


# --------------------------------------------------------------------------
# Resolved state
# --------------------------------------------------------------------------


@dataclass
class InputState:
    """Pointer and keyboard state for one specific frame."""

    x: float = 0.0
    y: float = 0.0
    click_x: float = 0.0
    click_y: float = 0.0
    button_down: bool = False
    #: Frame the current press began, or None if never pressed.
    press_frame: int | None = None
    #: True when coordinates are fractions of the resolution.
    normalized: bool = False
    held: frozenset[int] = frozenset()
    pressed: frozenset[int] = frozenset()
    toggled: frozenset[int] = frozenset()

    def mouse_vec4(self, width: int, height: int, frame: int) -> tuple[float, ...]:
        """The ``iMouse`` value for this frame."""
        scale_x, scale_y = (width, height) if self.normalized else (1.0, 1.0)
        x = self.x * scale_x
        y = self.y * scale_y
        cx = abs(self.click_x * scale_x)
        cy = abs(self.click_y * scale_y)
        if not self.button_down:
            # Released: both negative, magnitudes = where the last press was.
            return (x, y, -cx, -cy)
        is_press_frame = self.press_frame is not None and frame == self.press_frame
        return (x, y, cx, cy if is_press_frame else -cy)

    def keyboard_bytes(self) -> bytes:
        """The 256x3 keyboard texture for this frame."""
        data = bytearray(KEYBOARD_WIDTH * KEYBOARD_HEIGHT)
        for code in self.held:
            data[code] = 255
        for code in self.pressed:
            data[KEYBOARD_WIDTH + code] = 255
        for code in self.toggled:
            data[2 * KEYBOARD_WIDTH + code] = 255
        return bytes(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": [self.x, self.y],
            "click": [self.click_x, self.click_y],
            "button_down": self.button_down,
            "press_frame": self.press_frame,
            "normalized": self.normalized,
            "held": sorted(self.held),
            "pressed": sorted(self.pressed),
            "toggled": sorted(self.toggled),
        }


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


@dataclass
class InputTimeline:
    """An ordered list of operations, evaluable at any frame.

    Operations need not be written in temporal order -- scheduling a keypress
    next to the mouse drag it accompanies is often clearer than interleaving
    everything by frame. Order is therefore established here as a class
    invariant, not merely in the JSON parser, because :meth:`state_at` stops
    scanning at the first event beyond the requested frame; an unsorted list
    would silently yield wrong state rather than fail.

    Sorting is *stable*, so operations sharing a frame execute in the order they
    were written. That matters: ``mouse_down`` then ``mouse_move`` on one frame
    anchors the click where the press happened, whereas the reverse anchors it at
    the new position.
    """

    events: tuple[InputEvent, ...] = ()
    fps: float = 60.0

    def __post_init__(self) -> None:
        self.events = tuple(sorted(self.events, key=lambda e: e.frame))

    @classmethod
    def empty(cls, fps: float = 60.0) -> "InputTimeline":
        return cls(events=(), fps=fps)

    @classmethod
    def from_spec(cls, data: Any, fps: float = 60.0) -> "InputTimeline":
        """Build from a decoded JSON array of operations."""
        if data is None:
            return cls.empty(fps)
        if isinstance(data, dict):
            # A lone operation is accepted; requiring [ ] for one event is noise.
            data = [data]
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise InputError(
                f"input must be an array of operations, got {type(data).__name__}"
            )
        events = [parse_event(spec, fps, i) for i, spec in enumerate(data)]
        # __post_init__ sorts; the file order of same-frame events is preserved.
        return cls(events=tuple(events), fps=fps)

    @classmethod
    def from_json(cls, text: str, fps: float = 60.0) -> "InputTimeline":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(
                f"invalid input JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        return cls.from_spec(data, fps)

    @property
    def active(self) -> bool:
        return bool(self.events)

    @property
    def last_frame(self) -> int:
        return max((e.frame for e in self.events), default=0)

    def state_at(self, frame: int) -> InputState:
        """Replay every operation up to *frame* and return the resulting state."""
        x = y = 0.0
        click_x = click_y = 0.0
        normalized = False
        button_down = False
        press_frame: int | None = None
        held: set[int] = set()
        toggled: set[int] = set()
        pressed: set[int] = set()

        for event in self.events:
            if event.frame > frame:
                break
            if event.op in MOUSE_OPS:
                if event.pos is not None:
                    x, y = event.pos
                    normalized = event.normalized
                if event.op == "mouse_down":
                    button_down = True
                    click_x, click_y = x, y
                    press_frame = event.frame
                elif event.op == "mouse_up":
                    button_down = False
            elif event.op == "key_down":
                held.update(event.keys)
            elif event.op == "key_up":
                held.difference_update(event.keys)
            elif event.op == "key_tap":
                # Held for exactly the frame it occurs on.
                if event.frame == frame:
                    held.update(event.keys)
            elif event.op == "key_toggle":
                toggled.symmetric_difference_update(event.keys)
            elif event.op == "key_untoggle":
                toggled.difference_update(event.keys)

            # Row 1 is "pressed on this frame" and nothing else.
            if event.frame == frame and event.op in ("key_down", "key_tap"):
                pressed.update(event.keys)

        return InputState(
            x=x,
            y=y,
            click_x=click_x,
            click_y=click_y,
            button_down=button_down,
            press_frame=press_frame,
            normalized=normalized,
            held=frozenset(held),
            pressed=frozenset(pressed),
            toggled=frozenset(toggled),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "events": [e.to_dict() for e in self.events],
            "last_frame": self.last_frame,
        }


def load_input_spec(source: str, fps: float = 60.0) -> InputTimeline:
    """Load a timeline from inline JSON, a file path, or ``-`` for stdin."""
    text = source.strip()
    if text == "-":
        return InputTimeline.from_json(sys.stdin.read(), fps)
    if text.startswith("[") or text.startswith("{"):
        return InputTimeline.from_json(text, fps)
    path = Path(source).expanduser()
    if not path.is_file():
        raise InputError(
            f"input spec {source!r} is neither inline JSON (starting with '[') "
            f"nor an existing file"
        )
    try:
        return InputTimeline.from_json(path.read_text(encoding="utf-8"), fps)
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not valid UTF-8: {exc}") from exc
