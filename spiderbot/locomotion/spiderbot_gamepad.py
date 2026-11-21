"""
Gamepad for driving the robot: genesis-forge's `Gamepad`, plus a deadman and the SDL mapping
for the SpiderBot Bluetooth controller (hardware/controller/).

- Stick axes read 0.0 unless the deadman (SDL `lefttrigger`) is held, and everything drops to
  0.0 when the controller disconnects. A command is never held after release or link loss.
- The SpiderBot controller gets its SDL mapping as soon as SDL sees it (controller_mapping.py),
  replacing any mapping SDL picked automatically.
- Any other SDL game controller still works; hold LT as the deadman.

Usage:
    gamepad = SpiderBotGamepad()
    env.vel_command_manager.use_gamepad(gamepad, **DRIVE_AXES)
"""

from __future__ import annotations

import threading

import sdl2
from genesis_forge.gamepads import Gamepad
from genesis_forge.gamepads.sdl2 import ControllerEventLoop

from controller_mapping import apply_spiderbot_mapping

# SDL game-controller axis indices
AXIS_LEFTX = sdl2.SDL_CONTROLLER_AXIS_LEFTX     # 0
AXIS_LEFTY = sdl2.SDL_CONTROLLER_AXIS_LEFTY     # 1
AXIS_RIGHTX = sdl2.SDL_CONTROLLER_AXIS_RIGHTX   # 2

# Thumb stick: walk (Y) and turn (X). Index-finger stick: side-step (X).
# VelocityCommandManager.use_gamepad inverts all three, which is the right sign for these.
DRIVE_AXES = {
    "lin_vel_x_axis": AXIS_LEFTY,
    "ang_vel_z_axis": AXIS_LEFTX,
    "lin_vel_y_axis": AXIS_RIGHTX,
}

# Triggers aren't gated by the deadman (the deadman *is* a trigger).
UNGATED_AXES = (sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT, sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT)

# Raw SDL trigger values are 0..32767; held = more than half way. (Read raw because
# genesis-forge rescales a released trigger to 0.5, not 0.0.)
DEADMAN_THRESHOLD = 16384


class _EventLoop(ControllerEventLoop):
    """genesis-forge's SDL event loop, plus: map the SpiderBot controller when it appears,
    track the raw deadman value, and release everything when the last controller goes."""

    def __init__(self, gamepad: SpiderBotGamepad, **kwargs):
        super().__init__(**kwargs)
        self._gamepad = gamepad

    def _open_controller(self, device_index: int) -> None:
        # Mapping a device can make SDL emit CONTROLLERDEVICEADDED as well; open it once.
        if int(sdl2.SDL_JoystickGetDeviceInstanceID(device_index)) in self._controllers:
            return
        super()._open_controller(device_index)
        name = sdl2.SDL_GameControllerNameForIndex(device_index)
        print(f"🎮 Controller connected: {name.decode() if name else device_index}")

    def _close_controller(self, instance_id: int) -> None:
        # Also runs for every open controller when the loop exits, even on an exception.
        super()._close_controller(instance_id)
        if not self._controllers:
            self._gamepad.release_all()

    def _handle_event(self, event: sdl2.SDL_Event) -> None:
        if event.type == sdl2.SDL_JOYDEVICEADDED:
            # Arrives before CONTROLLERDEVICEADDED, so our mapping is in place first.
            if apply_spiderbot_mapping(event.jdevice.which):
                self._open_controller(event.jdevice.which)
            return
        if (
            event.type == sdl2.SDL_CONTROLLERAXISMOTION
            and event.caxis.axis == sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT
        ):
            self._gamepad.set_deadman(event.caxis.value)
        super()._handle_event(event)


class SpiderBotGamepad(Gamepad):
    """Drop-in replacement for `genesis_forge.gamepads.Gamepad` with a deadman."""

    def __init__(self):
        # Gamepad.__init__ hard-wires its own event loop, so this repeats its setup with ours.
        # Mirrors genesis-forge 0.4.0; re-check after upgrading it.
        self._debug = False
        self.is_running = True
        self._lock = threading.Lock()
        self._axis_values: list[float] = [0.0] * 6
        self._button_set: set[str] = set()
        self._dpad: str | None = None
        self._deadman_raw = 0

        self._event_loop = _EventLoop(self, handle_key=self._handle_key, alive=threading.Event())
        self._event_loop.alive.set()
        self._read_thread = threading.Thread(target=self._event_loop.run, daemon=True)
        self._read_thread.start()

    @property
    def deadman(self) -> bool:
        return self._deadman_raw > DEADMAN_THRESHOLD

    def axis(self, index: int) -> float:
        """Like Gamepad.axis, but stick axes read 0.0 unless the deadman is held."""
        if index not in UNGATED_AXES and not self.deadman:
            return 0.0
        return super().axis(index)

    def set_deadman(self, raw: int) -> None:
        """Raw SDL lefttrigger value, 0..32767. Called from the event-loop thread."""
        self._deadman_raw = raw

    def release_all(self) -> None:
        """Controller gone: zero everything."""
        with self._lock:
            self._axis_values = [0.0] * len(self._axis_values)
            self._button_set.clear()
            self._dpad = None
            self._deadman_raw = 0
        print("🎮 Controller disconnected: command zeroed")
