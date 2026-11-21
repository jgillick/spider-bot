#!/usr/bin/env python3
"""
Sanity-check the controller from SDL's point of view — the same API genesis-forge uses.

    pip install pysdl2 pysdl2-dll
    ./sdl_check.py

Prints whether SDL sees a *game controller* (not just a joystick) and streams the mapped
axes/buttons. For the SpiderBot controller you should see:
    leftx/lefty   = thumb stick
    rightx/righty = index-finger stick
    lefttrigger   = 0.0 / 1.0 with the deadman
    leftstick     = stick press
"""
import sys
import time
from pathlib import Path

import sdl2

# The mapping lives with the robot code, so this check uses exactly what play.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "spiderbot" / "locomotion"))
from controller_mapping import (  # noqa: E402
    apply_spiderbot_mapping,
    is_spiderbot_controller,
    joystick_guid,
    joystick_name,
)

AXES = {
    sdl2.SDL_CONTROLLER_AXIS_LEFTX: "leftx", sdl2.SDL_CONTROLLER_AXIS_LEFTY: "lefty",
    sdl2.SDL_CONTROLLER_AXIS_RIGHTX: "rightx", sdl2.SDL_CONTROLLER_AXIS_RIGHTY: "righty",
    sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT: "lefttrigger", sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT: "righttrigger",
}


def pick_controller(n: int) -> int | None:
    """The SpiderBot controller if present (adding its mapping), else any game controller."""
    for i in range(n):
        if is_spiderbot_controller(i) and apply_spiderbot_mapping(i):
            print("applied SpiderBot mapping")
            return i
    return next((i for i in range(n) if sdl2.SDL_IsGameController(i)), None)


def main() -> int:
    sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER)
    n = sdl2.SDL_NumJoysticks()
    if n == 0:
        print("SDL sees no joysticks. Is the controller connected (bluetoothctl info <addr>)?")
        return 1
    for i in range(n):
        is_gc = bool(sdl2.SDL_IsGameController(i))
        print(f"[{i}] {joystick_name(i)!r} guid={joystick_guid(i)} game_controller={is_gc}")
    idx = pick_controller(n)
    if idx is None:
        print("No device is recognised as a game controller; genesis-forge will not open it.")
        return 1
    gc = sdl2.SDL_GameControllerOpen(idx)
    print("mapping:", sdl2.SDL_GameControllerMapping(gc).decode())
    print("streaming (ctrl-c to stop)…")
    try:
        while True:
            sdl2.SDL_GameControllerUpdate()
            vals = []
            for a, nm in AXES.items():
                raw = sdl2.SDL_GameControllerGetAxis(gc, a)
                vals.append(f"{nm}={raw / 32767:+.2f}")
            ls = sdl2.SDL_GameControllerGetButton(gc, sdl2.SDL_CONTROLLER_BUTTON_LEFTSTICK)
            vals.append(f"leftstick={ls}")
            print("\r" + "  ".join(vals) + "   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        sdl2.SDL_GameControllerClose(gc)
        sdl2.SDL_Quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
