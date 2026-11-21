# SpiderBot Bluetooth controller

BLE HID gamepad (TinyS3[D], CircuitPython) for driving the robot through SDL / genesis-forge.

- [DESIGN.md](DESIGN.md): hardware, pin map, SDL mapping, behaviour spec, GATT protocol.
- [firmware/](firmware/): CircuitPython firmware, install and pairing steps.
- [tools/](tools/): `sdl_check.py` (what SDL sees), `robot_status.py` (GATT side channel: robot status
  to the controller, controller battery back, owner registration).
- [docs/](docs/): schematic (as wired).

Host side: `spiderbot/locomotion/spiderbot_gamepad.py` (deadman, stick layout) and `controller_mapping.py`
(SDL mapping), used by `play.py` and `play_motors.py`.

Tests (host-side, no hardware): `spiderbot/tests/controller/`. They cover the deadman and
zero-on-disconnect, the SDL mapping, stick scaling, saved settings, LED patterns, and the GATT
packet layouts.

```bash
uv run --with pytest python -m pytest spiderbot/tests/controller
```
