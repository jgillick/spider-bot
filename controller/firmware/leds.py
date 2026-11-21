"""Pairing / charging LEDs and the optional robot-battery bar."""
import digitalio

import config
from protocol import FLAG_FAULT, STATE_ADVERTISING, STATE_PAIRING

# Blink timings (ms)
PAIRING_BLINK_MS = 100      # pairing mode: fast blink
ADVERTISING_PERIOD_MS = 2000
ADVERTISING_WINK_MS = 60    # advertising: short wink once per period
FAULT_BLINK_MS = 150        # robot fault: whole bar flashes
LOW_BATTERY_BLINK_MS = 500
CHASE_STEP_MS = 120         # calibration sweep animation

# Robot battery % -> bar segments lit
BAR_THREE_ABOVE_PCT = 66
BAR_TWO_ABOVE_PCT = 33
BAR_BLINK_BELOW_PCT = 10


def make_output(pin, value=False):
    d = digitalio.DigitalInOut(pin)
    d.switch_to_output(value=value)
    return d


def _blink(now, period_ms):
    """True for the first half of every 2 * period_ms."""
    return (now // period_ms) % 2 == 0


class Leds:
    def __init__(self):
        self.pair = make_output(config.PIN_LED_PAIR)
        self.charge = make_output(config.PIN_LED_CHARGE)
        self.bar = [make_output(p) for p in config.PIN_LED_BAR]
        # Calibration feedback uses the bar if one is fitted, else pairing + charging.
        self._cal_leds = self.bar or [self.pair, self.charge]

    def all_off(self):
        self.pair.value = False
        self.charge.value = False
        self.set_bar(0)

    def show_pairing(self, link_state, now):
        if link_state == STATE_PAIRING:
            on = _blink(now, PAIRING_BLINK_MS)
        elif link_state == STATE_ADVERTISING:
            on = now % ADVERTISING_PERIOD_MS < ADVERTISING_WINK_MS
        else:
            on = False
        self.pair.value = on

    def show_charging(self, charging):
        self.charge.value = charging

    def set_bar(self, segments):
        for i, seg in enumerate(self.bar):
            seg.value = i < segments

    def show_robot_battery(self, robot_pct, robot_flags, fresh, now):
        """robot_pct: 0-100 or None. fresh: the robot reported recently over a live link."""
        if not fresh:
            self.set_bar(0)
        elif robot_flags & FLAG_FAULT:
            self.set_bar(len(self.bar) if _blink(now, FAULT_BLINK_MS) else 0)
        elif robot_pct is None:
            self.set_bar(0)
        elif robot_pct < BAR_BLINK_BELOW_PCT and not _blink(now, LOW_BATTERY_BLINK_MS):
            self.set_bar(0)
        elif robot_pct > BAR_THREE_ABOVE_PCT:
            self.set_bar(3)
        elif robot_pct > BAR_TWO_ABOVE_PCT:
            self.set_bar(2)
        else:
            self.set_bar(1)

    # --- calibration feedback ---
    def cal_all(self, on):
        for led in self._cal_leds:
            led.value = on

    def cal_chase(self, now):
        leds = self._cal_leds
        step = (now // CHASE_STEP_MS) % (len(leds) + 1)
        for i, led in enumerate(leds):
            led.value = i == step
