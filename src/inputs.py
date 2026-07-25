"""Simulated mouse and keyboard input.

Shadertoy shaders are interactive, so a test harness that cannot press a key or
drag the mouse can only exercise a fraction of a shader. Both devices are
modelled exactly as Shadertoy exposes them, and both are driven from CLI flags
so a run is fully reproducible.

Mouse (``iMouse``)
    ``xy`` is the current cursor position in pixels. ``zw`` encodes the position
    where the click began, with the button state folded into the signs:

    ==========================  ==========================
    ``iMouse.z > 0``            button is currently held
    ``iMouse.w > 0``            this is the frame of the press
    both negative               button released (``abs`` = last click)
    ==========================  ==========================

Keyboard (``iChannelN`` bound to ``keyboard``)
    A 256x3 texture read with ``texelFetch``:

    ====================================  ==============================
    ``texelFetch(ch, ivec2(code,0),0).x``  1.0 while the key is held
    ``texelFetch(ch, ivec2(code,1),0).x``  1.0 on the frame it was pressed
    ``texelFetch(ch, ivec2(code,2),0).x``  toggle, flips on each press
    ====================================  ==============================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

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

_BUTTON_STATES = ("up", "down", "click")


class InputError(ValueError):
    """Raised for malformed input specifications."""


def parse_key(token: str) -> int:
    """Resolve a key name or numeric code to a JavaScript key code."""
    token = token.strip()
    if not token:
        raise InputError("empty key specification")
    lowered = token.lower()
    if lowered in KEY_CODES:
        return KEY_CODES[lowered]
    if token.isdigit():
        code = int(token)
        # A bare digit is ambiguous: "5" as a name is the digit key (53), but a
        # raw code is also plausible. KEY_CODES already claimed 0-9 as names,
        # so anything reaching here is an explicit code.
        if not 0 <= code < KEYBOARD_WIDTH:
            raise InputError(
                f"key code {code} out of range 0..{KEYBOARD_WIDTH - 1}"
            )
        return code
    raise InputError(
        f"unknown key {token!r}. Use a name (a-z, 0-9, left, space, f1, ...) "
        f"or a numeric JavaScript key code."
    )


def parse_keys(tokens: Iterable[str]) -> list[int]:
    """Parse repeated and/or comma-separated key specs into codes."""
    codes: list[int] = []
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if part:
                codes.append(parse_key(part))
    return codes


@dataclass
class KeyboardState:
    """Which keys are held, pressed this frame, and toggled on."""

    held: set[int] = field(default_factory=set)
    pressed: set[int] = field(default_factory=set)
    toggled: set[int] = field(default_factory=set)

    @classmethod
    def from_spec(
        cls,
        keys: Iterable[str] = (),
        press: Iterable[str] = (),
        toggle: Iterable[str] = (),
    ) -> "KeyboardState":
        held = set(parse_keys(keys))
        pressed = set(parse_keys(press))
        # A key that is "pressed" this frame is necessarily also held.
        held |= pressed
        return cls(held=held, pressed=pressed, toggled=set(parse_keys(toggle)))

    @property
    def active(self) -> bool:
        return bool(self.held or self.pressed or self.toggled)

    def texture_bytes(self, frame: int = 0) -> bytes:
        """Build the 256x3 single-channel keyboard texture.

        ``pressed`` is only true on the first simulated frame; holding a key for
        many frames must not look like many separate presses.
        """
        data = bytearray(KEYBOARD_WIDTH * KEYBOARD_HEIGHT)
        for code in self.held:
            data[code] = 255
        if frame == 0:
            for code in self.pressed:
                data[KEYBOARD_WIDTH + code] = 255
        for code in self.toggled:
            data[2 * KEYBOARD_WIDTH + code] = 255
        return bytes(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "held": sorted(self.held),
            "pressed": sorted(self.pressed),
            "toggled": sorted(self.toggled),
        }


@dataclass
class MouseState:
    """Cursor position and button state, encoded the way Shadertoy does."""

    x: float = 0.0
    y: float = 0.0
    click_x: float | None = None
    click_y: float | None = None
    button: str = "up"
    #: When set, coordinates are fractions of the resolution rather than pixels.
    normalized: bool = False

    @classmethod
    def from_spec(
        cls,
        position: str | None = None,
        button: str | None = None,
        click: str | None = None,
        normalized: bool = False,
    ) -> "MouseState":
        x = y = 0.0
        if position is not None:
            x, y = _parse_pair(position, "--mouse")
        cx = cy = None
        if click is not None:
            cx, cy = _parse_pair(click, "--mouse-click")
        if button is None:
            # Supplying a position implies the button is held; that is the only
            # state in which most shaders read iMouse at all.
            button = "down" if position is not None or click is not None else "up"
        button = button.lower()
        if button not in _BUTTON_STATES:
            raise InputError(
                f"mouse button must be one of {', '.join(_BUTTON_STATES)} "
                f"(got {button!r})"
            )
        if position is None and click is not None:
            x, y = cx, cy
        return cls(
            x=x, y=y, click_x=cx, click_y=cy, button=button, normalized=normalized
        )

    @property
    def active(self) -> bool:
        return self.button != "up" or (self.x, self.y) != (0.0, 0.0)

    def as_vec4(self, width: int, height: int, frame: int = 0) -> tuple[float, ...]:
        """Resolve to the ``iMouse`` value for a given resolution and frame."""
        scale_x, scale_y = (width, height) if self.normalized else (1.0, 1.0)
        x = self.x * scale_x
        y = self.y * scale_y
        cx = self.click_x if self.click_x is not None else self.x
        cy = self.click_y if self.click_y is not None else self.y
        cx *= scale_x
        cy *= scale_y

        if self.button == "up":
            # Released: both components negative, magnitudes = last click.
            return (x, y, -abs(cx), -abs(cy))
        # A click is only "new" on the first frame of the run.
        is_press_frame = self.button == "click" and frame == 0
        return (x, y, abs(cx), abs(cy) if is_press_frame else -abs(cy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "click_x": self.click_x,
            "click_y": self.click_y,
            "button": self.button,
            "normalized": self.normalized,
        }


def _parse_pair(text: str, flag: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
    if len(parts) != 2:
        raise InputError(f"{flag} expects 'X,Y' (got {text!r})")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise InputError(f"{flag} expects two numbers (got {text!r})") from None
