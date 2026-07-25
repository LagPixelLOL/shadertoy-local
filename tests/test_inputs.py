"""Simulated mouse and keyboard state."""

from __future__ import annotations

import pytest

from shadertoy_local.inputs import (
    KEYBOARD_WIDTH,
    InputError,
    KeyboardState,
    MouseState,
    parse_key,
    parse_keys,
)


class TestKeyParsing:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("space", 32), ("a", 65), ("z", 90), ("A", 65),
            ("left", 37), ("up", 38), ("right", 39), ("down", 40),
            ("enter", 13), ("escape", 27), ("esc", 27),
            ("f1", 112), ("f12", 123),
            ("numpad0", 96),
            ("5", 53),          # digit key, not raw code 5
            ("200", 200),       # out of name range -> raw code
        ],
    )
    def test_named_and_numeric(self, token, expected):
        assert parse_key(token) == expected

    def test_unknown_key(self):
        with pytest.raises(InputError, match="unknown key"):
            parse_key("hyperspace")

    def test_empty_key(self):
        with pytest.raises(InputError, match="empty key"):
            parse_key("  ")

    def test_out_of_range_code(self):
        with pytest.raises(InputError, match="out of range"):
            parse_key(str(KEYBOARD_WIDTH + 1))

    def test_comma_separated_and_repeated(self):
        assert parse_keys(["w,a", "s", " d "]) == [87, 65, 83, 68]


class TestKeyboardTexture:
    def _rows(self, state: KeyboardState, frame: int = 0):
        data = state.texture_bytes(frame)
        assert len(data) == KEYBOARD_WIDTH * 3
        return (
            data[:KEYBOARD_WIDTH],
            data[KEYBOARD_WIDTH : 2 * KEYBOARD_WIDTH],
            data[2 * KEYBOARD_WIDTH :],
        )

    def test_held_sets_row_zero_only(self):
        held, pressed, toggled = self._rows(KeyboardState.from_spec(keys=["space"]))
        assert held[32] == 255
        assert pressed[32] == 0
        assert toggled[32] == 0

    def test_press_implies_held(self):
        held, pressed, _ = self._rows(KeyboardState.from_spec(press=["space"]))
        assert held[32] == 255
        assert pressed[32] == 255

    def test_press_only_fires_on_first_frame(self):
        """Holding a key for many frames must not look like many presses."""
        state = KeyboardState.from_spec(press=["space"])
        _, pressed_f0, _ = self._rows(state, frame=0)
        held_f5, pressed_f5, _ = self._rows(state, frame=5)
        assert pressed_f0[32] == 255
        assert pressed_f5[32] == 0
        assert held_f5[32] == 255

    def test_toggle_sets_row_two(self):
        _, _, toggled = self._rows(KeyboardState.from_spec(toggle=["g"]))
        assert toggled[71] == 255

    def test_nothing_pressed_is_all_zero(self):
        assert set(KeyboardState().texture_bytes(0)) == {0}

    def test_active_flag(self):
        assert not KeyboardState().active
        assert KeyboardState.from_spec(keys=["a"]).active


class TestMouseEncoding:
    def test_default_is_released(self):
        x, y, z, w = MouseState().as_vec4(640, 360)
        assert (x, y) == (0.0, 0.0)
        assert z <= 0 and w <= 0

    def test_position_implies_button_down(self):
        state = MouseState.from_spec(position="320,180")
        assert state.button == "down"
        x, y, z, w = state.as_vec4(640, 360)
        assert (x, y) == (320.0, 180.0)
        assert z > 0, "iMouse.z must be positive while held"

    def test_explicit_up_clears_z(self):
        state = MouseState.from_spec(position="320,180", button="up")
        _, _, z, _ = state.as_vec4(640, 360)
        assert z <= 0

    def test_click_frame_has_positive_w(self):
        state = MouseState.from_spec(position="10,20", button="click")
        _, _, z, w = state.as_vec4(640, 360, frame=0)
        assert z > 0 and w > 0
        # On later frames it is merely held, not newly clicked.
        _, _, z2, w2 = state.as_vec4(640, 360, frame=1)
        assert z2 > 0 and w2 < 0

    def test_separate_click_origin(self):
        state = MouseState.from_spec(position="100,100", click="50,60")
        x, y, z, w = state.as_vec4(640, 360)
        assert (x, y) == (100.0, 100.0)
        assert (abs(z), abs(w)) == (50.0, 60.0)

    def test_normalized_coordinates_scale(self):
        state = MouseState.from_spec(position="0.5,0.25", normalized=True)
        x, y, _, _ = state.as_vec4(640, 360)
        assert (x, y) == (320.0, 90.0)

    def test_bad_pair(self):
        with pytest.raises(InputError, match="expects 'X,Y'"):
            MouseState.from_spec(position="320")

    def test_non_numeric(self):
        with pytest.raises(InputError, match="two numbers"):
            MouseState.from_spec(position="a,b")

    def test_bad_button(self):
        with pytest.raises(InputError, match="must be one of"):
            MouseState.from_spec(position="1,2", button="wiggle")
