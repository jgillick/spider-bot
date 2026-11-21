"""Values shared between the firmware and the host over BLE. See ../DESIGN.md.

tools/robot_status.py uses the same numbers on the host side; keep the two in step.
"""

# Link state: what the pairing LED shows, and the `state` byte of controller-status.
STATE_IDLE = 0
STATE_ADVERTISING = 1
STATE_CONNECTED = 2
STATE_PAIRING = 3

# Battery percentage meaning "unknown" (no fuel gauge, or the robot didn't say).
BATTERY_UNKNOWN = 255

# Robot-status `flags` byte.
FLAG_FAULT = 0x01

# HID report button bits.
BUTTON_L3 = 0x01
BUTTON_TRIGGER = 0x02
