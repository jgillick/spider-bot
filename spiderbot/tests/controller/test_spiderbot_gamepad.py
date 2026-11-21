"""
Tests for spiderbot/locomotion/spiderbot_gamepad.py and controller_mapping.py.

The deadman and zero-on-disconnect behaviour is what stops the robot when the operator lets
go or the controller drops out, so it's pinned down here. No SDL device or thread needed.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("sdl2")
pytest.importorskip("genesis_forge")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "locomotion"))

import sdl2  # noqa: E402

import controller_mapping  # noqa: E402
from spiderbot_gamepad import (  # noqa: E402
    DEADMAN_THRESHOLD,
    DRIVE_AXES,
    SpiderBotGamepad,
    _EventLoop,
)

TRIGGER_LEFT = sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT
TRIGGER_RIGHT = sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT


def make_gamepad(axis_values=(0.5, -0.5, 0.25, 0.0, 1.0, 0.0)):
    """A SpiderBotGamepad with its state set directly; no SDL thread."""
    with patch("spiderbot_gamepad.threading.Thread"), patch("spiderbot_gamepad._EventLoop"):
        gp = SpiderBotGamepad()
    gp._axis_values = list(axis_values)
    return gp


# ---------------------------------------------------------------------------
# deadman gating
# ---------------------------------------------------------------------------


def test_sticks_read_zero_without_deadman():
    gp = make_gamepad()
    assert [gp.axis(i) for i in range(4)] == [0.0, 0.0, 0.0, 0.0]


def test_sticks_pass_through_with_deadman():
    gp = make_gamepad()
    gp.set_deadman(32767)
    assert [gp.axis(i) for i in range(4)] == [0.5, -0.5, 0.25, 0.0]


def test_deadman_threshold_is_exclusive():
    gp = make_gamepad()
    gp.set_deadman(DEADMAN_THRESHOLD)
    assert not gp.deadman
    assert gp.axis(0) == 0.0
    gp.set_deadman(DEADMAN_THRESHOLD + 1)
    assert gp.deadman
    assert gp.axis(0) == 0.5


@pytest.mark.parametrize("axis", [TRIGGER_LEFT, TRIGGER_RIGHT])
def test_triggers_are_not_gated(axis):
    gp = make_gamepad()
    assert gp.axis(axis) == gp._axis_values[axis]


def test_drive_axes_are_the_stick_layout():
    assert DRIVE_AXES == {"lin_vel_x_axis": 1, "ang_vel_z_axis": 0, "lin_vel_y_axis": 2}


# ---------------------------------------------------------------------------
# release on disconnect
# ---------------------------------------------------------------------------


def test_release_all_zeroes_everything():
    gp = make_gamepad()
    gp.set_deadman(32767)
    gp._button_set.add("a")
    gp._dpad = "up"

    gp.release_all()

    assert gp._axis_values == [0.0] * 6
    assert gp._button_set == set()
    assert gp._dpad is None
    assert not gp.deadman
    assert gp.axis(0) == 0.0


def make_loop(controllers):
    loop = _EventLoop.__new__(_EventLoop)
    loop._controllers = dict(controllers)
    loop._gamepad = MagicMock()
    return loop


def fake_parent_close(self, instance_id):
    self._controllers.pop(instance_id, None)


def test_closing_last_controller_releases_all():
    loop = make_loop({7: object()})
    with patch("genesis_forge.gamepads.sdl2.ControllerEventLoop._close_controller", fake_parent_close):
        loop._close_controller(7)
    loop._gamepad.release_all.assert_called_once()


def test_closing_one_of_two_controllers_keeps_state():
    loop = make_loop({7: object(), 8: object()})
    with patch("genesis_forge.gamepads.sdl2.ControllerEventLoop._close_controller", fake_parent_close):
        loop._close_controller(7)
    loop._gamepad.release_all.assert_not_called()


def test_trigger_motion_updates_deadman():
    loop = make_loop({})
    event = MagicMock()
    event.type = sdl2.SDL_CONTROLLERAXISMOTION
    event.caxis.axis = TRIGGER_LEFT
    event.caxis.value = 30000
    with patch("genesis_forge.gamepads.sdl2.ControllerEventLoop._handle_event"):
        loop._handle_event(event)
    loop._gamepad.set_deadman.assert_called_once_with(30000)


# ---------------------------------------------------------------------------
# SDL mapping
# ---------------------------------------------------------------------------


def fake_sdl(name, is_controller_after=True, add_result=1):
    sdl = MagicMock()
    sdl.SDL_JoystickNameForIndex.return_value = name
    sdl.SDL_GameControllerAddMapping.return_value = add_result
    sdl.SDL_IsGameController.return_value = is_controller_after
    return sdl


def test_mapping_ignores_other_devices():
    sdl = fake_sdl(b"Xbox Wireless Controller")
    with patch.object(controller_mapping, "sdl2", sdl):
        assert controller_mapping.apply_spiderbot_mapping(0) is False
    sdl.SDL_GameControllerAddMapping.assert_not_called()


def test_mapping_handles_nameless_device():
    sdl = fake_sdl(None)
    with patch.object(controller_mapping, "sdl2", sdl):
        assert controller_mapping.apply_spiderbot_mapping(0) is False


def test_mapping_applied_even_if_sdl_already_maps_it():
    sdl = fake_sdl(b"SpiderBot Controller")
    with patch.object(controller_mapping, "sdl2", sdl), patch.object(
        controller_mapping, "joystick_guid", return_value="0500abcd"
    ):
        assert controller_mapping.apply_spiderbot_mapping(3) is True
    sdl.SDL_GameControllerAddMapping.assert_called_once_with(
        b"0500abcd,SpiderBot Controller,"
        b"leftx:a0,lefty:a1,rightx:a2,righty:a3,leftstick:b0,lefttrigger:b1,"
    )
