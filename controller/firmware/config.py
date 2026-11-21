"""Pins and tuning for the SpiderBot Bluetooth controller (TinyS3[D], CircuitPython 10.x).

Everything you might need to change after wiring or hand-testing lives here.
Pin map follows the schematic in ../docs/schematic.svg (as wired). Nothing else in the
firmware hard-codes a pin.
"""
import board

# --- pins ---------------------------------------------------------------------------
PIN_LX = board.IO1        # thumb stick X    (ADC1, A0)
PIN_LY = board.IO2        # thumb stick Y    (ADC1, A1)
PIN_RX = board.IO3        # index stick X    (ADC1, A2)
PIN_RY = board.IO4        # index stick Y    (ADC1, A3)
PIN_POT_PWR = board.IO35  # 3V3 to all four pots; high while awake

PIN_L3 = board.IO5        # thumb stick press, active low
PIN_TRIGGER = board.IO6   # deadman key, active low
PIN_POWER = board.IO7     # slide switch, closed/low = ON (RTC IO: deep-sleep wake)

PIN_LED_PAIR = board.IO36    # blue
PIN_LED_CHARGE = board.IO37  # amber
# Robot-battery bar, bottom -> top. Not on the schematic; add pins here if you fit one.
# With no bar, the robot's battery isn't shown and calibration uses the two LEDs above.
PIN_LED_BAR = ()

PIN_VBUS = board.VBUS_SENSE      # IO33, high when USB 5 V is present
PIN_NEOPIXEL_PWR = board.NEOPIXEL_POWER   # held low to save power

# --- sticks -------------------------------------------------------------------------
OVERSAMPLE = 16          # ADC reads averaged per axis per loop
DEADZONE = 0.02          # fraction of full scale around centre
# Flip an axis if it reads backwards. Convention to aim for (Xbox/SDL):
#   X: right = positive.   Y: forward/up = NEGATIVE.
INVERT = (False, False, False, False)   # (LX, LY, RX, RY)
# Used until a calibration is stored: half-range from centre, in raw 16-bit ADC counts.
# Extents widen automatically as the sticks are pushed further than this.
DEFAULT_HALF_RANGE = 20000
# If the centre captured at power-on is further than this from the stored centre
# (fraction of the stored half-range), assume a thumb was on the stick and keep the stored one.
CENTRE_SANITY = 0.15

# --- timing (ms) ----------------------------------------------------------------------
LOOP_MS = 10                  # ~100 Hz
REPORT_KEEPALIVE_MS = 50      # resend unchanged report this often
REPORT_CHANGE_COUNTS = 48     # axis change (of +-32767) that counts as "changed"
DEBOUNCE_MS = 20
PAIR_GESTURE_MS = 4000        # TRIGGER + L3 held
PAIR_WINDOW_MS = 60000        # pairing mode times out after this
CAL_GESTURE_MS = 3000         # L3 held at switch-on
CAL_SWEEP_MS = 5000
CAL_MIN_TRAVEL = 4000         # raw ADC counts each side of centre a sweep must cover to be saved
AUTH_TIMEOUT_MS = 10000       # drop links that haven't authenticated by now
AUTH_TIMEOUT_PAIRING_MS = 30000   # ...longer while a human is confirming the code on the host
ROBOT_STATUS_TIMEOUT_MS = 5000
GAUGE_PERIOD_MS = 5000
GAUGE_PERIOD_OFF_MS = 2000
CHARGE_FULL_PCT = 98

# --- BLE ------------------------------------------------------------------------------
BLE_NAME = "SpiderBot Controller"
# After bonding, advertise only to the stored owner address (set by the host over GATT).
# Needs a host with a fixed public address (the Pi). Leave False for macOS hosts.
DIRECTED_ADVERTISING = True
