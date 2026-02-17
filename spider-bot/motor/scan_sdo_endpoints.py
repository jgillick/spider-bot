#!/usr/bin/env python3
"""
Scan SDO endpoints: try to read each as a float and print the value or None.
Use this to validate endpoint IDs (e.g. encoder offset, position estimate) for your motor.
"""

import sys

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

from odrive_can import ODriveController

# Configuration
BITRATE = 500000
NODE_ID = 1
ENDPOINT_MIN = 0
ENDPOINT_MAX = 500
READ_TIMEOUT_S = 0.4

IGNORE_PORTS = ["/dev/cu.debug-console", "/dev/cu.Bluetooth-Incoming-Port"]


def choose_port():
    """List serial ports and let user select one."""
    if list_ports is None:
        print("pyserial required for port listing. Install with: pip install pyserial")
        port = input("Enter serial port path (e.g. /dev/cu.usbmodem1101): ").strip()
        return port or None

    ports = list(list_ports.comports())
    ports = [p for p in ports if p.device not in IGNORE_PORTS]
    if not ports:
        print("No serial ports found.")
        port = input("Enter serial port path manually: ").strip()
        return port or None

    print("\nAvailable serial ports:")
    for i, p in enumerate(ports, 1):
        desc = p.description or "—"
        print(f"  {i}) {p.device}  —  {desc}")
    print(f"  {len(ports) + 1}) Enter path manually")

    while True:
        try:
            choice = input("\nSelect port [1]: ").strip() or "1"
            n = int(choice)
            if 1 <= n <= len(ports):
                return ports[n - 1].device
            if n == len(ports) + 1:
                return input("Enter port path: ").strip() or None
        except ValueError:
            pass
        print("Invalid choice.")


def main():
    port = choose_port()
    if not port:
        print("No port selected. Exiting.")
        sys.exit(1)

    print(f"\nConnecting to motor on {port} (node_id={NODE_ID})...")
    print(
        f"Scanning SDO endpoints {ENDPOINT_MIN}–{ENDPOINT_MAX} (timeout={READ_TIMEOUT_S}s each).\n"
    )

    with ODriveController(NODE_ID, port=port, bitrate=BITRATE) as motor:
        for endpoint_id in range(ENDPOINT_MIN, ENDPOINT_MAX + 1):
            try:
                value = motor.rxsdo_read_float(endpoint_id, timeout_s=READ_TIMEOUT_S)
                print(f"  {endpoint_id}: {value}")
            except TimeoutError:
                pass
            except Exception as e:
                print(f"  {endpoint_id}: None  ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
