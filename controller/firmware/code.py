"""SpiderBot Bluetooth controller firmware (Unexpected Maker TinyS3[D], CircuitPython 10.x).

Behaviour spec: ../DESIGN.md. Pins and tuning: config.py.

  power switch off              -> off state (charge LED only on USB, else deep sleep until on)
  L3 held at switch-on for 3 s  -> stick calibration (release, hands off, then sweep for 5 s)
  TRIGGER + L3 held for 4 s     -> erase bonds, pairing mode for 60 s

Files: sticks.py (analog sticks), ble_link.py (Bluetooth), leds.py, store.py (saved settings),
protocol.py (values shared with the host).
"""
import time
import traceback

import alarm
import board
import digitalio
import supervisor
from adafruit_ticks import ticks_add, ticks_diff, ticks_ms

import config
from ble_link import Link
from leds import Leds, make_output
from protocol import BUTTON_L3, BUTTON_TRIGGER
from sticks import Sticks
from store import Store

ZERO_AXES = (0, 0, 0, 0)


class Button:
    """Active-low input (pressed = pin pulled to ground) with a time-based debounce."""

    def __init__(self, pin):
        self._io = digitalio.DigitalInOut(pin)
        self._io.switch_to_input(pull=digitalio.Pull.UP)
        self.pressed = self.is_down()          # debounced state, updated by update()
        self._last_raw = self.pressed
        self._changed_at = ticks_ms()

    def is_down(self):
        """Raw, un-debounced state. For blocking waits outside the main loop."""
        return not self._io.value

    def update(self, now):
        """Call every tick. Returns the debounced state."""
        raw = self.is_down()
        if raw != self._last_raw:
            self._last_raw = raw
            self._changed_at = now
        elif raw != self.pressed and ticks_diff(now, self._changed_at) >= config.DEBOUNCE_MS:
            self.pressed = raw
        return self.pressed

    def deinit(self):
        self._io.deinit()


class HoldGesture:
    """Fires once when its buttons have all been held for `hold_ms`; re-arms on release."""

    def __init__(self, hold_ms):
        self._hold_ms = hold_ms
        self._since = None
        self._fired = False
        self.held = False   # all buttons down right now

    def update(self, all_down, now):
        """Returns True on the tick the hold completes."""
        self.held = all_down
        if not all_down:
            self._since = None
            self._fired = False
            return False
        if self._since is None:
            self._since = now
        if not self._fired and ticks_diff(now, self._since) >= self._hold_ms:
            self._fired = True
            return True
        return False


class ReportPacer:
    """Decides when to send a HID report: on a real change, or as a keep-alive."""

    def __init__(self):
        self._axes = [0, 0, 0, 0]
        self._buttons = 0
        self._sent_at = ticks_ms()
        self._force = True       # send the first report regardless

    def due(self, axes, buttons, now):
        if self._force or buttons != self._buttons:
            return True
        if ticks_diff(now, self._sent_at) >= config.REPORT_KEEPALIVE_MS:
            return True
        for i in range(4):
            if abs(axes[i] - self._axes[i]) > config.REPORT_CHANGE_COUNTS:
                return True
        return False

    def sent(self, axes, buttons, now):
        for i in range(4):
            self._axes[i] = axes[i]
        self._buttons = buttons
        self._sent_at = now
        self._force = False


class Gauge:
    """The MAX17048 fuel gauge on the TinyS3[D].

    read() returns (percent 0-100, millivolts), or (None, 0) if the gauge isn't answering.
    """

    def __init__(self):
        self._dev = None
        try:
            import adafruit_max1704x
            self._dev = adafruit_max1704x.MAX17048(board.I2C())
        except Exception as e:
            print("no fuel gauge:", e)

    def read(self):
        if self._dev is None:
            return None, 0
        try:
            pct = max(0, min(100, int(self._dev.cell_percent + 0.5)))
            mv = int(self._dev.cell_voltage * 1000)
            return pct, mv
        except Exception as e:
            print("gauge:", e)
            return None, 0


def is_charging(usb_present, battery_pct):
    return bool(usb_present and battery_pct is not None and battery_pct < config.CHARGE_FULL_PCT)


def off_state(power, vbus, leds, gauge):
    """Never returns: reloads code.py when switched on, or deep-sleeps until it is."""
    leds.all_off()
    print("off")
    next_gauge = ticks_ms()
    while vbus.value:
        now = ticks_ms()
        if power.update(now):
            supervisor.reload()
        if ticks_diff(now, next_gauge) >= 0:
            pct, _ = gauge.read()
            leds.show_charging(is_charging(True, pct))
            next_gauge = ticks_add(now, config.GAUGE_PERIOD_OFF_MS)
        time.sleep(0.05)
    # On battery: deep sleep, wake when the switch closes. (IO33/VBUS is not an RTC pin on
    # the S3, so plugging in USB can't wake us; the charger still charges regardless.)
    leds.all_off()
    power.deinit()
    wake = alarm.pin.PinAlarm(config.PIN_POWER, value=False, pull=True)
    alarm.exit_and_deep_sleep_until_alarms(wake)


def calibration_requested(l3):
    """True if L3 is held down for CAL_GESTURE_MS right after switch-on."""
    if not l3.is_down():
        return False
    until = ticks_add(ticks_ms(), config.CAL_GESTURE_MS)
    while l3.is_down() and ticks_diff(until, ticks_ms()) > 0:
        time.sleep(0.01)
    return l3.is_down()


def calibrate(sticks, l3, leds, store):
    """Release L3, hands off for 1 s, then sweep both sticks around their full travel."""
    leds.cal_all(True)
    print("calibration: release L3 and let go of the sticks")
    while l3.is_down():
        time.sleep(0.01)
    time.sleep(1.0)
    sticks.begin_sweep()
    print("calibration: sweep both sticks to every edge")
    end = ticks_add(ticks_ms(), config.CAL_SWEEP_MS)
    while ticks_diff(end, ticks_ms()) > 0:
        sticks.sweep_step()
        leds.cal_chase(ticks_ms())

    cal = sticks.calibration()
    travel = config.CAL_MIN_TRAVEL
    ok = all(hi - centre > travel and centre - lo > travel for lo, centre, hi in cal)
    print("calibration:", cal, "saved" if ok else "rejected (an axis barely moved)")
    if ok:
        store.save_cal(cal)

    flashes, period = (3, 0.3) if ok else (6, 0.1)   # 3 slow = saved, 6 fast = rejected
    for _ in range(flashes):
        leds.cal_all(True)
        time.sleep(period)
        leds.cal_all(False)
        time.sleep(period)


def main():
    try:
        supervisor.disable_ble_workflow()   # keep the BLE file-transfer service out of the way
    except (AttributeError, NotImplementedError):
        pass

    make_output(config.PIN_NEOPIXEL_PWR, False)
    leds = Leds()
    vbus = digitalio.DigitalInOut(config.PIN_VBUS)
    vbus.switch_to_input()
    power = Button(config.PIN_POWER)
    gauge = Gauge()
    store = Store()

    if not power.pressed:
        off_state(power, vbus, leds, gauge)

    # --- boot ------------------------------------------------------------------------
    pot_pwr = make_output(config.PIN_POT_PWR, True)
    time.sleep(0.02)                 # let the pots settle before the first ADC read
    sticks = Sticks()
    l3 = Button(config.PIN_L3)
    trigger = Button(config.PIN_TRIGGER)

    if calibration_requested(l3):
        calibrate(sticks, l3, leds, store)
        time.sleep(1.0)              # hands off again before centres are measured
    sticks.start(store.cal)

    link = Link(store)
    pairing_gesture = HoldGesture(config.PAIR_GESTURE_MS)
    pacer = ReportPacer()
    print("ready:", config.BLE_NAME, "owner stored" if store.owner else "no owner")

    # --- run ---------------------------------------------------------------------------
    next_gauge = ticks_ms()
    next_tick = ticks_ms()
    while True:
        now = ticks_ms()

        if not power.update(now):
            link.shutdown()
            pot_pwr.value = False
            off_state(power, vbus, leds, gauge)

        l3.update(now)
        trigger.update(now)
        axes = sticks.read()
        buttons = (BUTTON_L3 if l3.pressed else 0) | (BUTTON_TRIGGER if trigger.pressed else 0)

        # TRIGGER + L3: outputs are zero while both are down; after 4 s, pairing mode.
        if pairing_gesture.update(l3.pressed and trigger.pressed, now):
            link.enter_pairing(now)
        if pairing_gesture.held:
            axes, buttons = ZERO_AXES, 0

        link.poll(now)

        if link.connected and pacer.due(axes, buttons, now):
            link.send(axes, buttons)
            pacer.sent(axes, buttons, now)

        if ticks_diff(now, next_gauge) >= 0:
            pct, mv = gauge.read()
            charging = is_charging(vbus.value, pct)
            link.publish_battery(pct, charging, mv)
            leds.show_charging(charging)
            next_gauge = ticks_add(now, config.GAUGE_PERIOD_MS)

        leds.show_pairing(link.state, now)
        leds.show_robot_battery(
            link.robot_pct, link.robot_flags, link.robot_status_fresh(now), now
        )

        next_tick = ticks_add(next_tick, config.LOOP_MS)
        wait = ticks_diff(next_tick, ticks_ms())
        if wait > 0:
            time.sleep(wait / 1000)
        else:
            next_tick = ticks_ms()       # fell behind; don't try to catch up


# An unexpected error would otherwise stop the controller dead until it's power-cycled.
# Print it and restart instead. (Reload, deep sleep and Ctrl-C aren't Exceptions, so they
# still work normally.)
try:
    main()
except Exception as e:
    traceback.print_exception(e)
    time.sleep(2)                    # don't spin if it fails straight away again
    supervisor.reload()
