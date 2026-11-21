"""
SDL game-controller mapping for the SpiderBot Bluetooth controller (hardware/controller/).

The controller is a generic BLE HID gamepad, so SDL has no built-in mapping for it. Its HID
report (firmware ble_link.GAMEPAD_DESCRIPTOR) is: X, Y, Z, Rz axes, then button 1 = L3 and
button 2 = deadman. Shared by spiderbot_gamepad.py and hardware/controller/tools/sdl_check.py.
Only depends on pysdl2.
"""

from __future__ import annotations

import ctypes

import sdl2

CONTROLLER_NAME = "SpiderBot Controller"
MAPPING = "leftx:a0,lefty:a1,rightx:a2,righty:a3,leftstick:b0,lefttrigger:b1,"


def joystick_name(device_index: int) -> str:
    raw = sdl2.SDL_JoystickNameForIndex(device_index)
    return raw.decode(errors="replace") if raw else ""


def joystick_guid(device_index: int) -> str:
    buf = ctypes.create_string_buffer(64)
    sdl2.SDL_JoystickGetGUIDString(sdl2.SDL_JoystickGetDeviceGUID(device_index), buf, 64)
    return buf.value.decode()


def is_spiderbot_controller(device_index: int) -> bool:
    return CONTROLLER_NAME in joystick_name(device_index)


def apply_spiderbot_mapping(device_index: int) -> bool:
    """If `device_index` is a SpiderBot controller, make SDL use our mapping for it.

    Always applied, even if SDL already treats the device as a game controller: recent SDL
    versions on Linux auto-map generic HID gamepads, and that automatic mapping doesn't put
    the deadman on `lefttrigger`. Re-applying is harmless (SDL just updates the mapping).

    Returns True if it is a SpiderBot controller and SDL can now open it as a game controller.
    """
    if not is_spiderbot_controller(device_index):
        return False
    mapping = f"{joystick_guid(device_index)},{CONTROLLER_NAME},{MAPPING}"
    if sdl2.SDL_GameControllerAddMapping(mapping.encode()) < 0:
        print("SDL rejected the SpiderBot mapping:", sdl2.SDL_GetError().decode())
        return False
    return bool(sdl2.SDL_IsGameController(device_index))
