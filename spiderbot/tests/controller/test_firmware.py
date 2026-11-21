"""
Tests for the pure logic in hardware/controller/firmware (stick scaling, saved settings, LED
patterns), run on the host with the CircuitPython hardware modules stubbed out.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

FIRMWARE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "hardware", "controller", "firmware"
)

# CircuitPython-only modules the firmware imports.
NVM = bytearray(8192)
sys.modules.setdefault("board", MagicMock())
sys.modules.setdefault("analogio", MagicMock())
sys.modules.setdefault("digitalio", MagicMock())
sys.modules.setdefault("microcontroller", types.SimpleNamespace(nvm=NVM))
sys.path.insert(0, FIRMWARE)

import config  # noqa: E402
import leds  # noqa: E402
import store  # noqa: E402
from protocol import FLAG_FAULT, STATE_ADVERTISING, STATE_CONNECTED, STATE_PAIRING  # noqa: E402
from sticks import REPORT_MAX, Axis, Sticks  # noqa: E402


# ---------------------------------------------------------------------------
# sticks
# ---------------------------------------------------------------------------


def make_axis(lo=10000, centre=30000, hi=60000, invert=False):
    axis = Axis(pin=None, invert=invert)
    axis.set_calibration(lo, centre, hi)
    return axis


def test_centre_reads_zero():
    assert make_axis().to_report(30000) == 0


def test_inside_deadzone_reads_zero():
    axis = make_axis()
    just_inside = 30000 + int(config.DEADZONE * 30000) - 1
    assert axis.to_report(just_inside) == 0


def test_full_travel_reaches_full_scale_both_ways():
    # Asymmetric halves: centre is not mid-travel, both ends still reach full scale.
    axis = make_axis(lo=10000, centre=30000, hi=60000)
    assert axis.to_report(60000) == REPORT_MAX
    assert axis.to_report(10000) == -REPORT_MAX


def test_travel_past_calibration_widens_and_clamps():
    axis = make_axis(hi=60000)
    assert axis.to_report(65000) == REPORT_MAX
    assert axis.hi == 65000


def test_invert_flips_sign():
    assert make_axis(invert=True).to_report(60000) == -REPORT_MAX


def test_collapsed_calibration_does_not_divide_by_zero():
    axis = make_axis(lo=30000, centre=30000, hi=30000)
    assert axis.to_report(30000) == 0


def make_sticks(raw_centres):
    sticks = Sticks()
    for axis, raw in zip(sticks.axes, raw_centres):
        axis.read_raw = lambda raw=raw: raw
    return sticks


def test_start_without_calibration_uses_default_half_range():
    sticks = make_sticks([30000] * 4)
    sticks.start(None)
    half = config.DEFAULT_HALF_RANGE
    assert sticks.calibration()[0] == (30000 - half, 30000, 30000 + half)


def test_start_keeps_stored_centre_when_stick_is_touched():
    stored = [(10000, 30000, 50000)] * 4
    sticks = make_sticks([45000, 30500, 30000, 30000])   # axis 0 far off, axis 1 close
    sticks.start(stored)
    cal = sticks.calibration()
    assert cal[0] == (10000, 30000, 50000)   # stored centre kept
    assert cal[1] == (10000, 30500, 50000)   # fresh centre used


# ---------------------------------------------------------------------------
# saved settings
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_nvm():
    NVM[:] = bytes(len(NVM))


def test_store_round_trip(blank_nvm):
    s = store.Store()
    cal = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    s.save_cal(cal)
    s.save_owner(b"\x01\x02\x03\x04\x05\x06", 0)

    loaded = store.Store()
    assert loaded.cal == cal
    assert loaded.owner == (b"\x01\x02\x03\x04\x05\x06", 0)


def test_store_ignores_blank_nvm(blank_nvm):
    s = store.Store()
    assert s.cal is None
    assert s.owner is None


def test_clear_owner_keeps_calibration(blank_nvm):
    s = store.Store()
    s.save_cal([(1, 2, 3)] * 4)
    s.save_owner(b"\xaa" * 6, 1)
    s.clear_owner()

    loaded = store.Store()
    assert loaded.owner is None
    assert loaded.cal == [(1, 2, 3)] * 4


# ---------------------------------------------------------------------------
# LEDs
# ---------------------------------------------------------------------------


class Pin:
    value = False


def make_leds(bar_segments=3):
    l = leds.Leds.__new__(leds.Leds)
    l.pair, l.charge = Pin(), Pin()
    l.bar = [Pin() for _ in range(bar_segments)]
    l._cal_leds = l.bar or [l.pair, l.charge]
    return l


def lit(l):
    return sum(seg.value for seg in l.bar)


ON_PHASE = 0   # ms value where every blink is in its "on" half


@pytest.mark.parametrize("pct, segments", [(100, 3), (67, 3), (66, 2), (34, 2), (33, 1), (10, 1)])
def test_bar_segments(pct, segments):
    l = make_leds()
    l.show_robot_battery(pct, 0, True, ON_PHASE)
    assert lit(l) == segments


def test_bar_dark_when_status_is_stale():
    l = make_leds()
    l.show_robot_battery(100, FLAG_FAULT, False, ON_PHASE)
    assert lit(l) == 0


def test_bar_dark_when_battery_unknown():
    l = make_leds()
    l.show_robot_battery(None, 0, True, ON_PHASE)
    assert lit(l) == 0


def test_bar_flashes_fully_on_fault():
    l = make_leds()
    l.show_robot_battery(None, FLAG_FAULT, True, ON_PHASE)
    assert lit(l) == 3
    l.show_robot_battery(None, FLAG_FAULT, True, leds.FAULT_BLINK_MS)
    assert lit(l) == 0


def test_bar_blinks_below_ten_percent():
    l = make_leds()
    l.show_robot_battery(5, 0, True, ON_PHASE)
    assert lit(l) == 1
    l.show_robot_battery(5, 0, True, leds.LOW_BATTERY_BLINK_MS)
    assert lit(l) == 0


def test_no_bar_fitted_is_harmless():
    l = make_leds(bar_segments=0)
    l.show_robot_battery(100, FLAG_FAULT, True, ON_PHASE)
    l.cal_chase(0)
    assert l._cal_leds == [l.pair, l.charge]


def test_pairing_led_patterns():
    l = make_leds()
    l.show_pairing(STATE_PAIRING, ON_PHASE)
    assert l.pair.value
    l.show_pairing(STATE_ADVERTISING, leds.ADVERTISING_WINK_MS + 1)
    assert not l.pair.value
    l.show_pairing(STATE_CONNECTED, ON_PHASE)
    assert not l.pair.value
