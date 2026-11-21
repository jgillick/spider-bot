# Controller firmware (CircuitPython)

Firmware for the SpiderBot Bluetooth controller described in [../DESIGN.md](../DESIGN.md).

## Install

1. Flash a current **CircuitPython 10.x** build for **Unexpected Maker TinyS3[D]**
   (`unexpectedmaker_tinys3d`). Older builds don't have numeric-comparison pairing or `directed_to`.
2. Install the libraries:

   ```bash
   pip install circup
   circup install adafruit_ble adafruit_max1704x adafruit_ticks
   ```

3. Copy `code.py`, `config.py`, `protocol.py`, `ble_link.py`, `sticks.py`, `leds.py` and `store.py`
   to `CIRCUITPY/`.

Serial console (`tio /dev/tty.usbmodem*` or the Mu serial pane) prints link and pairing events.

## Using it

| what | how |
|---|---|
| On / off | slide switch. Off on battery = deep sleep; off on USB = only the charge LED runs. |
| Pair a new host | hold **TRIGGER + L3** for 4 s. Bonds are erased, the pairing LED blinks fast, and the controller accepts one pairing for 60 s. Stick output is zero while the two are held. |
| Calibrate sticks | hold **L3** while switching on, for 3 s. Both LEDs light: let go of everything, wait 1 s, then sweep both sticks around their full range while the LEDs alternate (5 s). 3 slow flashes = saved, 6 fast = rejected. |

LEDs:

- **pairing (blue):** fast blink = pairing mode, short wink every 2 s = advertising, off = connected.
- **charging (amber):** on while USB is present and the battery is under 98 %.
- **robot-battery bar:** not wired on this build. If you add one, list its pins in
  `config.PIN_LED_BAR`: 3/2/1 segments at >66/>33/≤33 %, blinking under 10 %, flashing = robot
  fault, dark = no robot status for 5 s or no link. Calibration then uses the bar too.

Stick centres are re-captured at every switch-on, so keep thumbs off the sticks while switching on
(if a centre is far from the stored calibration, the stored one is used instead).
If a stick reads backwards, flip it in `config.INVERT`. The target convention is right = +X, forward = −Y.

## Pairing with the Pi (one time)

```text
sudo sed -i 's/^#\?AutoEnable=.*/AutoEnable=true/' /etc/bluetooth/main.conf
bluetoothctl
  agent DisplayYesNo
  default-agent
  scan on                      # now hold TRIGGER + L3 for 4 s on the controller
  pair <addr>                  # answer "yes" to the confirm prompt
  trust <addr>
  connect <addr>
```

Then tell the controller who its owner is, so it only advertises to the Pi from now on:

```bash
python ../tools/robot_status.py --set-owner
```

Until an owner is stored (and always when `config.DIRECTED_ADVERTISING = False`), the controller
advertises normally, but still refuses to pair outside pairing mode. Use undirected advertising
for macOS hosts, whose Bluetooth addresses rotate.

## How the pairing lock works

Every HID and GATT attribute requires `LESC_ENCRYPT_WITH_MITM`. That makes every new pairing a
numeric comparison, which the firmware answers itself: **accept** in pairing mode, **reject**
otherwise. Bonded hosts reconnect with their stored keys and skip the prompt. Any link that
hasn't authenticated within 10 s (30 s in pairing mode) is dropped, which also catches "Just Works"
peers. Only one authenticated host is kept.

## Not verified on hardware yet

This was written against the CircuitPython 10.x `_bleio` source and the `adafruit_ble` library,
but hasn't been run on the board. Things to check first:

- `tools/sdl_check.py` on the Pi shows the device, and the SDL mapping (leftx/lefty/rightx/righty
  = a0..a3, leftstick = b0, lefttrigger = b1) puts each control where you expect.
- ADC loop time: 4 axes × 16 samples should fit inside the 10 ms loop. If not, lower `OVERSAMPLE`.
- Directed advertising reconnects the Pi after a power cycle. If it doesn't, set
  `DIRECTED_ADVERTISING = False`.
