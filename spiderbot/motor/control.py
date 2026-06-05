#!/usr/bin/env python3
"""
Interactive motor position controller CLI.
Uses CanMotorController for single-node motor control via CAN bus.
"""

import sys
import time
import select
import termios
import tty
import shutil
from typing import Optional

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

from .can_motor_controller import CanMotorController
from .protocol import AxisState, ControlMode, InputMode

# Configuration
BITRATE = 500000
NODE_ID = 1

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


def _enable(bus: CanMotorController, node_id: int) -> None:
    """Enable closed-loop position control on *node_id*."""
    bus.send_clear_errors(node_id)
    time.sleep(0.05)
    bus.send_set_control_mode(node_id, ControlMode.POSITION_CONTROL, InputMode.PASSTHROUGH)
    time.sleep(0.05)
    bus.send_set_state(node_id, AxisState.CLOSED_LOOP_CONTROL)
    time.sleep(0.05)


def run_controller_loop(bus: CanMotorController, node_id: int, port: str) -> str:
    """
    Run the main control loop.  Returns ``'quit'`` or ``'reconnect'``.
    """
    old_settings = termios.tcgetattr(sys.stdin)

    # Virtual zero offset (radians, absolute encoder frame).
    zero_offset: float = 0.0
    target_position: Optional[float] = None  # radians, relative to virtual zero

    try:
        tty.setcbreak(sys.stdin.fileno())

        print("\n" + "=" * 50)
        print("Motor Position Controller")
        print("=" * 50)
        print("Commands:")
        print("  Type a number and press Enter to set position")
        print("  'e' - Enable closed loop control")
        print("  'i' - Set to idle (disable)")
        print("  'z' - Set current position as zero (software)")
        print("  'Z' - Save current position as zero to motor (persistent)")
        print("  'h' - Go to zero (home)")
        print("  'q' - Quit")
        print("=" * 50 + "\n")

        input_buffer = ""
        last_update = 0

        while True:
            state = bus.get_state(node_id)
            position = state.position - zero_offset

            if time.time() - last_update > 0.1:
                target_str = (
                    f"{target_position:.3f}"
                    if target_position is not None
                    else "---"
                )
                status = (
                    f"\rPos: {position:8.3f} rad | "
                    f"Vel: {state.velocity:7.3f} rad/s | "
                    f"State: {state.axis_state.name:12s} | "
                    f"Target: {target_str:>8s} | "
                    f"Input: {input_buffer:10s}"
                )
                try:
                    width = shutil.get_terminal_size().columns
                except OSError:
                    width = 80
                sys.stdout.write(status[:width].ljust(width))
                sys.stdout.flush()
                last_update = time.time()

            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)

                if char == "q":
                    print("\n\nQuitting...")
                    return "quit"

                elif char == "e":
                    print("\n\nEnabling closed loop control...")
                    _enable(bus, node_id)
                    input_buffer = ""
                    time.sleep(0.5)

                elif char == "i":
                    print("\n\nDisabling motor...")
                    bus.send_set_state(node_id, AxisState.IDLE)
                    input_buffer = ""

                elif char == "z":
                    print("\n\nSetting current position as zero (software)...")
                    zero_offset = state.position
                    target_position = 0.0
                    input_buffer = ""

                elif char == "Z":
                    print("\n\nSaving current position as zero to motor flash...")
                    bus.save_zero_to_motor(node_id)
                    input_buffer = ""
                    return "reconnect"

                elif char == "h":
                    print("\n\nGoing to zero (home)...")
                    target_position = 0.0
                    bus.send_position(node_id, zero_offset)
                    input_buffer = ""

                elif char == "\n" or char == "\r":
                    if input_buffer:
                        try:
                            pos = float(input_buffer)
                            print(f"\n\nCommanding position: {pos} rad")
                            # motor.move_to(pos)
                            bus.send_mit_control(node_id, pos)
                            target_position = pos
                        except ValueError:
                            print(f"\n\nInvalid input: {input_buffer}")
                    input_buffer = ""

                elif char == "\x7f" or char == "\x08":
                    input_buffer = input_buffer[:-1]

                elif char in "0123456789.-":
                    input_buffer += char

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        return "quit"
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    port = choose_port()
    if not port:
        print("No port selected. Exiting.")
        sys.exit(1)

    while True:
        print(f"\nConnecting to motor on {port}...")
        with CanMotorController(channel=port, bitrate=BITRATE) as bus:
            bus.register_node(NODE_ID)
            result = run_controller_loop(bus, NODE_ID, port)

        if result == "quit":
            break
        if result == "reconnect":
            print("Motor rebooting. Reconnecting in 3 seconds...")
            time.sleep(3)

    print("Closed cleanly")


if __name__ == "__main__":
    main()
