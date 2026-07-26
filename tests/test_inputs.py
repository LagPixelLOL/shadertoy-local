"""Simulated input: the operation timeline."""

from __future__ import annotations

import json

import pytest

from shadertoy_local.inputs import (
    KEYBOARD_WIDTH,
    InputError,
    InputEvent,
    InputState,
    InputTimeline,
    load_input_spec,
    parse_key,
    parse_keys,
)


def timeline(ops, fps: float = 60.0) -> InputTimeline:
    return InputTimeline.from_spec(ops, fps)


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
            ("200", 200),       # beyond the name range -> raw code
            (200, 200),         # a JSON number is a raw code
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
            parse_key(KEYBOARD_WIDTH + 1)

    def test_booleans_rejected(self):
        """JSON true would otherwise silently become key code 1."""
        with pytest.raises(InputError, match="name or code"):
            parse_key(True)

    def test_list_and_comma_forms(self):
        assert parse_keys(["w,a", "s", " d "]) == [87, 65, 83, 68]
        assert parse_keys("w,a") == [87, 65]


class TestEventValidation:
    def test_requires_frame_or_time(self):
        with pytest.raises(InputError, match='needs a "frame" or "time"'):
            timeline([{"op": "mouse_up"}])

    def test_rejects_both_frame_and_time(self):
        with pytest.raises(InputError, match="not both"):
            timeline([{"frame": 1, "time": 1.0, "op": "mouse_up"}])

    def test_unknown_op(self):
        with pytest.raises(InputError, match='"op" must be one of'):
            timeline([{"frame": 0, "op": "wiggle"}])

    def test_unknown_key_in_object(self):
        with pytest.raises(InputError, match="unknown key"):
            timeline([{"frame": 0, "op": "mouse_up", "colour": "red"}])

    def test_negative_frame(self):
        with pytest.raises(InputError, match=">= 0"):
            timeline([{"frame": -1, "op": "mouse_up"}])

    def test_non_integer_frame(self):
        with pytest.raises(InputError, match="must be an integer"):
            timeline([{"frame": 1.5, "op": "mouse_up"}])

    def test_mouse_move_needs_pos(self):
        with pytest.raises(InputError, match='needs a "pos"'):
            timeline([{"frame": 0, "op": "mouse_move"}])

    def test_key_op_needs_keys(self):
        with pytest.raises(InputError, match='needs "keys"'):
            timeline([{"frame": 0, "op": "key_down"}])

    def test_pos_rejected_on_key_op(self):
        with pytest.raises(InputError, match='"pos" does not apply'):
            timeline([{"frame": 0, "op": "key_down", "keys": ["a"], "pos": [1, 2]}])

    def test_keys_rejected_on_mouse_op(self):
        with pytest.raises(InputError, match='"keys" does not apply'):
            timeline([{"frame": 0, "op": "mouse_up", "keys": ["a"]}])

    def test_bad_pos_shape(self):
        with pytest.raises(InputError, match="exactly two numbers"):
            timeline([{"frame": 0, "op": "mouse_move", "pos": [1, 2, 3]}])

    def test_empty_keys(self):
        with pytest.raises(InputError, match="is empty"):
            timeline([{"frame": 0, "op": "key_down", "keys": []}])

    def test_top_level_must_be_a_list(self):
        with pytest.raises(InputError, match="must be an array"):
            timeline("mouse_down")

    def test_single_object_is_accepted(self):
        """Requiring [ ] around one operation would be pointless ceremony."""
        line = timeline({"frame": 0, "op": "key_down", "keys": ["a"]})
        assert 65 in line.state_at(0).held

    def test_invalid_json_reports_position(self):
        with pytest.raises(InputError, match="invalid input JSON at line"):
            InputTimeline.from_json('[{"frame": ')


class TestTimeAndFrame:
    def test_time_converts_via_fps(self):
        line = timeline([{"time": 0.5, "op": "key_down", "keys": ["a"]}], fps=60.0)
        assert line.events[0].frame == 30

    def test_time_rounds_to_nearest(self):
        """0.5s at 60fps is frame 30, not 29 from truncation."""
        assert timeline([{"time": 0.5, "op": "mouse_up"}], fps=60.0).events[0].frame == 30
        assert timeline([{"time": 0.51, "op": "mouse_up"}], fps=60.0).events[0].frame == 31

    def test_fps_affects_conversion(self):
        assert timeline([{"time": 1.0, "op": "mouse_up"}], fps=30.0).events[0].frame == 30
        assert timeline([{"time": 1.0, "op": "mouse_up"}], fps=24.0).events[0].frame == 24

    def test_frame_and_time_mix_freely(self):
        line = timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [1, 1]},
                {"time": 1.0, "op": "mouse_up"},
            ],
            fps=60.0,
        )
        assert line.state_at(59).button_down is True
        assert line.state_at(60).button_down is False

    def test_negative_time(self):
        with pytest.raises(InputError, match=">= 0"):
            timeline([{"time": -0.5, "op": "mouse_up"}])

    def test_original_time_is_echoed(self):
        event = timeline([{"time": 0.25, "op": "mouse_up"}], fps=60.0).events[0]
        assert event.time == 0.25
        assert event.to_dict()["time"] == 0.25


class TestOrdering:
    """Operations need not be written in temporal order."""

    OPS = [
        {"frame": 20, "op": "mouse_up"},
        {"frame": 5, "op": "key_down", "keys": ["w"]},
        {"frame": 0, "op": "mouse_down", "pos": [10, 10]},
        {"frame": 10, "op": "mouse_move", "pos": [50, 50]},
    ]
    SORTED = [OPS[2], OPS[1], OPS[3], OPS[0]]

    def test_events_are_sorted_on_construction(self):
        assert [e.frame for e in timeline(self.OPS).events] == [0, 5, 10, 20]

    def test_shuffled_input_matches_ordered_input(self):
        shuffled = timeline(self.OPS)
        ordered = timeline(self.SORTED)
        for frame in range(0, 26):
            assert shuffled.state_at(frame) == ordered.state_at(frame), frame

    def test_direct_construction_is_also_sorted(self):
        """state_at stops at the first event past the frame, so an unsorted tuple
        would silently produce wrong state rather than fail."""
        line = InputTimeline(
            events=(
                InputEvent(frame=20, op="mouse_up"),
                InputEvent(frame=0, op="mouse_down", pos=(10.0, 10.0)),
            )
        )
        assert [e.frame for e in line.events] == [0, 20]
        assert line.state_at(5).button_down is True

    def test_same_frame_order_is_preserved(self):
        """Stable sort: down-then-move anchors the click at the press position,
        move-then-down anchors it at the moved position."""
        down_first = timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [10, 10]},
                {"frame": 0, "op": "mouse_move", "pos": [90, 90]},
            ]
        ).state_at(0)
        move_first = timeline(
            [
                {"frame": 0, "op": "mouse_move", "pos": [90, 90]},
                {"frame": 0, "op": "mouse_down", "pos": [10, 10]},
            ]
        ).state_at(0)
        assert (down_first.click_x, down_first.x) == (10.0, 90.0)
        assert (move_first.click_x, move_first.x) == (10.0, 10.0)

    def test_last_frame(self):
        assert timeline(self.OPS).last_frame == 20
        assert InputTimeline.empty().last_frame == 0


class TestMouseEvaluation:
    def test_empty_timeline_is_released_at_origin(self):
        state = InputTimeline.empty().state_at(0)
        x, y, z, w = state.mouse_vec4(640, 360, 0)
        assert (x, y) == (0.0, 0.0)
        assert z <= 0 and w <= 0

    def test_press_sets_z_positive(self):
        line = timeline([{"frame": 0, "op": "mouse_down", "pos": [320, 180]}])
        x, y, z, w = line.state_at(0).mouse_vec4(640, 360, 0)
        assert (x, y) == (320.0, 180.0)
        assert z > 0

    def test_w_positive_only_on_the_press_frame(self):
        line = timeline([{"frame": 7, "op": "mouse_down", "pos": [10, 20]}])
        _, _, z7, w7 = line.state_at(7).mouse_vec4(640, 360, 7)
        _, _, z8, w8 = line.state_at(8).mouse_vec4(640, 360, 8)
        assert z7 > 0 and w7 > 0, "press frame must have w > 0"
        assert z8 > 0 and w8 < 0, "later frames are held, not newly pressed"

    def test_release_makes_both_negative(self):
        line = timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [30, 40]},
                {"frame": 5, "op": "mouse_up"},
            ]
        )
        _, _, z, w = line.state_at(6).mouse_vec4(640, 360, 6)
        assert z < 0 and w < 0
        # Magnitudes still report where the last press was.
        assert (abs(z), abs(w)) == (30.0, 40.0)

    def test_drag_moves_cursor_but_keeps_click_anchor(self):
        line = timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [100, 100]},
                {"frame": 10, "op": "mouse_move", "pos": [200, 150]},
            ]
        )
        state = line.state_at(10)
        x, y, z, w = state.mouse_vec4(640, 360, 10)
        assert (x, y) == (200.0, 150.0)
        assert (abs(z), abs(w)) == (100.0, 100.0)

    def test_second_press_updates_the_anchor(self):
        line = timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [10, 10]},
                {"frame": 5, "op": "mouse_up"},
                {"frame": 9, "op": "mouse_down", "pos": [80, 90]},
            ]
        )
        state = line.state_at(9)
        assert state.press_frame == 9
        _, _, z, w = state.mouse_vec4(640, 360, 9)
        assert (abs(z), abs(w)) == (80.0, 90.0)
        assert w > 0

    def test_normalized_positions_scale(self):
        line = timeline(
            [{"frame": 0, "op": "mouse_move", "pos": [0.5, 0.25], "normalized": True}]
        )
        x, y, _, _ = line.state_at(0).mouse_vec4(640, 360, 0)
        assert (x, y) == (320.0, 90.0)

    def test_normalized_requires_pos(self):
        with pytest.raises(InputError, match='only applies alongside "pos"'):
            timeline([{"frame": 0, "op": "mouse_up", "normalized": True}])


class TestPointerBounds:
    """The pointer is confined to the canvas, as it is on shadertoy.com.

    The site attaches its listeners to the canvas element with no pointer
    capture, so an event outside it never fires and ``iMouse`` cannot hold an
    off-canvas value. Accepting one locally would let a shader be tuned against
    input that the site can never deliver.
    """

    def _at(self, pos, **extra):
        return [{"frame": 0, "op": "mouse_move", "pos": pos, **extra}]

    @pytest.mark.parametrize(
        "pos,axis",
        [
            ([69420, 180], "x"),
            ([320, 9999], "y"),
            ([-1, 180], "x"),
            ([320, -69.42], "y"),
            ([640.5, 180], "x"),
            ([320, 360.5], "y"),
        ],
    )
    def test_off_canvas_pixels_are_rejected(self, pos, axis):
        with pytest.raises(InputError, match=f"pos {axis}=.*outside the 640x360"):
            InputTimeline.from_spec(self._at(pos), 60.0, 640, 360)

    @pytest.mark.parametrize("pos", [[0, 0], [640, 360], [320.5, 180.25], [0, 360]])
    def test_on_canvas_pixels_are_accepted(self, pos):
        """Bounds are inclusive: the canvas spans [0, width] continuously."""
        line = InputTimeline.from_spec(self._at(pos), 60.0, 640, 360)
        assert line.events[0].pos == (float(pos[0]), float(pos[1]))

    def test_bounds_follow_the_actual_resolution(self):
        """200 is fine on a wide canvas and off a narrow one."""
        assert InputTimeline.from_spec(self._at([200, 20]), 60.0, 640, 360).active
        with pytest.raises(InputError, match=r"outside the 100x50 canvas \(0\.\.100\)"):
            InputTimeline.from_spec(self._at([200, 20]), 60.0, 100, 50)

    def test_negative_is_rejected_even_without_a_resolution(self):
        """No upper bound is knowable, but a negative coordinate is still wrong.

        iMouse.zw encodes the button state in its signs and mouse_vec4 takes
        abs() of the anchor, so a negative press position would be silently
        flipped positive instead of surviving to be noticed.
        """
        with pytest.raises(InputError, match="pos y=-5 is outside the canvas"):
            InputTimeline.from_spec(self._at([10, -5]))

    def test_large_positive_passes_without_a_resolution(self):
        """The library stays usable with no render target; the CLI supplies one."""
        assert InputTimeline.from_spec(self._at([69420, 180])).active

    @pytest.mark.parametrize("pos", [[1.5, 0.5], [0.5, -0.25], [0.5, 2]])
    def test_normalized_outside_unit_range_is_rejected(self, pos):
        with pytest.raises(InputError, match=r"normalized pos [xy]=.*outside the"):
            InputTimeline.from_spec(self._at(pos, normalized=True), 60.0, 640, 360)

    @pytest.mark.parametrize("pos", [[0, 0], [1, 1], [0.5, 0.25]])
    def test_normalized_unit_range_is_accepted(self, pos):
        line = InputTimeline.from_spec(self._at(pos, normalized=True), 60.0, 640, 360)
        assert line.events[0].normalized

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_is_rejected(self, bad):
        with pytest.raises(InputError, match="must be a finite number"):
            InputTimeline.from_spec(self._at([bad, 10]), 60.0, 640, 360)

    def test_string_form_is_bounds_checked_too(self):
        """"x,y" must not be a way around the check."""
        with pytest.raises(InputError, match="outside the 640x360"):
            InputTimeline.from_spec(self._at("700, 10"), 60.0, 640, 360)

    def test_click_anchor_is_bounds_checked(self):
        """mouse_down carries the anchor, so it matters most of all."""
        with pytest.raises(InputError, match="outside the 640x360"):
            InputTimeline.from_spec(
                [{"frame": 0, "op": "mouse_down", "pos": [-3, 10]}], 60.0, 640, 360
            )


class TestMixedPositionUnits:
    """One timeline, one unit.

    InputState carries a single ``normalized`` flag taken from the last
    positioned event, and mouse_vec4 applies it to the cursor and the click
    anchor alike -- so a pixel press followed by a normalized move used to
    rescale the already-pixel anchor by the resolution.
    """

    PIXEL_THEN_NORMALIZED = [
        {"frame": 0, "op": "mouse_down", "pos": [320, 180]},
        {"frame": 5, "op": "mouse_move", "pos": [0.5, 0.5], "normalized": True},
    ]

    def test_mixing_units_is_rejected(self):
        with pytest.raises(InputError, match="pixels but input.1. gives it normalized"):
            timeline(self.PIXEL_THEN_NORMALIZED)

    def test_the_anchor_corruption_it_prevents(self):
        """Construct the mixed state directly to show what the check averts.

        The parser now refuses this, so the only way to reach the state is to
        build the events by hand -- which pins the misbehaviour the rule exists
        for: a press at x=320 reported to the shader as x=204800.
        """
        line = InputTimeline(
            events=(
                InputEvent(frame=0, op="mouse_down", pos=(320.0, 180.0)),
                InputEvent(frame=5, op="mouse_move", pos=(0.5, 0.5), normalized=True),
            )
        )
        _, _, z, _ = line.state_at(5).mouse_vec4(640, 360, 5)
        assert z == 204800.0
        with pytest.raises(InputError):
            timeline(self.PIXEL_THEN_NORMALIZED)

    def test_normalized_then_pixel_is_rejected_too(self):
        with pytest.raises(InputError, match="One timeline must use one unit"):
            timeline(
                [
                    {"frame": 0, "op": "mouse_down", "pos": [0.5, 0.5],
                     "normalized": True},
                    {"frame": 5, "op": "mouse_move", "pos": [320, 180]},
                ]
            )

    def test_consistent_units_are_fine(self):
        assert timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [0.5, 0.5], "normalized": True},
                {"frame": 5, "op": "mouse_move", "pos": [0.75, 0.5],
                 "normalized": True},
            ]
        ).active

    def test_positionless_ops_do_not_count_as_a_unit(self):
        """mouse_up without a pos carries no coordinate to disagree about."""
        assert timeline(
            [
                {"frame": 0, "op": "mouse_down", "pos": [0.5, 0.5], "normalized": True},
                {"frame": 5, "op": "mouse_up"},
                {"frame": 6, "op": "key_down", "keys": ["w"]},
            ]
        ).active


class TestKeyboardEvaluation:
    def _rows(self, state: InputState):
        data = state.keyboard_bytes()
        assert len(data) == KEYBOARD_WIDTH * 3
        return (
            data[:KEYBOARD_WIDTH],
            data[KEYBOARD_WIDTH : 2 * KEYBOARD_WIDTH],
            data[2 * KEYBOARD_WIDTH :],
        )

    def test_empty_is_all_zero(self):
        assert set(InputTimeline.empty().state_at(0).keyboard_bytes()) == {0}

    def test_key_down_holds_until_released(self):
        line = timeline(
            [
                {"frame": 2, "op": "key_down", "keys": ["space"]},
                {"frame": 8, "op": "key_up", "keys": ["space"]},
            ]
        )
        assert self._rows(line.state_at(1))[0][32] == 0
        assert self._rows(line.state_at(2))[0][32] == 255
        assert self._rows(line.state_at(7))[0][32] == 255
        assert self._rows(line.state_at(8))[0][32] == 0

    def test_pressed_row_fires_only_on_the_event_frame(self):
        line = timeline([{"frame": 4, "op": "key_down", "keys": ["space"]}])
        assert self._rows(line.state_at(4))[1][32] == 255
        assert self._rows(line.state_at(5))[1][32] == 0
        # ...but it stays held.
        assert self._rows(line.state_at(5))[0][32] == 255

    def test_key_tap_lasts_one_frame(self):
        line = timeline([{"frame": 3, "op": "key_tap", "keys": ["g"]}])
        held3, pressed3, _ = self._rows(line.state_at(3))
        held4, pressed4, _ = self._rows(line.state_at(4))
        assert (held3[71], pressed3[71]) == (255, 255)
        assert (held4[71], pressed4[71]) == (0, 0)

    def test_toggle_flips_each_time(self):
        line = timeline(
            [
                {"frame": 1, "op": "key_toggle", "keys": ["g"]},
                {"frame": 5, "op": "key_toggle", "keys": ["g"]},
            ]
        )
        assert self._rows(line.state_at(0))[2][71] == 0
        assert self._rows(line.state_at(1))[2][71] == 255
        assert self._rows(line.state_at(5))[2][71] == 0

    def test_untoggle_clears(self):
        line = timeline(
            [
                {"frame": 1, "op": "key_toggle", "keys": ["g"]},
                {"frame": 3, "op": "key_untoggle", "keys": ["g"]},
            ]
        )
        assert self._rows(line.state_at(3))[2][71] == 0

    def test_multiple_keys_in_one_op(self):
        line = timeline([{"frame": 0, "op": "key_down", "keys": ["w", "a", "s", "d"]}])
        held, _, _ = self._rows(line.state_at(0))
        assert all(held[code] == 255 for code in (87, 65, 83, 68))

    def test_partial_release(self):
        line = timeline(
            [
                {"frame": 0, "op": "key_down", "keys": ["w", "a"]},
                {"frame": 5, "op": "key_up", "keys": ["w"]},
            ]
        )
        held, _, _ = self._rows(line.state_at(5))
        assert held[87] == 0 and held[65] == 255


class TestCombinedTimeline:
    """Pointer and keyboard in one stream, which is the whole point."""

    OPS = [
        {"frame": 0, "op": "mouse_down", "pos": [320, 180]},
        {"frame": 0, "op": "key_down", "keys": ["w"]},
        {"frame": 30, "op": "mouse_move", "pos": [500, 180]},
        {"time": 1.0, "op": "key_tap", "keys": ["space"]},
        {"frame": 60, "op": "mouse_up"},
        {"frame": 60, "op": "key_up", "keys": ["w"]},
    ]

    def test_states_across_the_run(self):
        line = timeline(self.OPS, fps=60.0)

        at0 = line.state_at(0)
        assert at0.button_down and 87 in at0.held and at0.press_frame == 0

        at30 = line.state_at(30)
        assert (at30.x, at30.y) == (500.0, 180.0)
        assert at30.button_down and 87 in at30.held

        # key_tap at time 1.0 -> frame 60
        at60 = line.state_at(60)
        assert 32 in at60.pressed and 32 in at60.held
        assert not at60.button_down, "mouse_up on frame 60"
        assert 87 not in at60.held, "key_up on frame 60"

        at61 = line.state_at(61)
        assert 32 not in at61.held, "tap lasts one frame"

    def test_serialisable(self):
        payload = timeline(self.OPS).to_dict()
        restored = json.loads(json.dumps(payload))
        assert restored["last_frame"] == 60
        assert len(restored["events"]) == 6

    def test_state_serialisable(self):
        state = timeline(self.OPS).state_at(30)
        assert json.loads(json.dumps(state.to_dict()))["button_down"] is True


class TestLoadInputSpec:
    def test_inline_json(self):
        line = load_input_spec('[{"frame": 0, "op": "key_down", "keys": ["a"]}]')
        assert 65 in line.state_at(0).held

    def test_file_path(self, tmp_path):
        path = tmp_path / "input.json"
        path.write_text('[{"frame": 2, "op": "key_down", "keys": ["b"]}]')
        line = load_input_spec(str(path))
        assert 66 in line.state_at(2).held

    def test_missing_file_is_a_clear_error(self):
        with pytest.raises(InputError, match="neither inline JSON"):
            load_input_spec("does-not-exist.json")

    def test_stdin(self, monkeypatch):
        import io

        monkeypatch.setattr(
            "sys.stdin", io.StringIO('[{"frame": 0, "op": "key_tap", "keys": ["x"]}]')
        )
        line = load_input_spec("-")
        assert 88 in line.state_at(0).pressed

    def test_fps_is_honoured(self, tmp_path):
        path = tmp_path / "i.json"
        path.write_text('[{"time": 1.0, "op": "mouse_up"}]')
        assert load_input_spec(str(path), fps=30.0).events[0].frame == 30
