"""
Shared setup for the real leg motors: CAN node mapping, rotation direction, and
the alignment-tool zero positions.
"""

import os
import json
import time
from serial.tools import list_ports
from spiderbot.motor.can_motor_controller import CanMotorController
from spiderbot.motor.protocol import AxisState, ControlMode, InputMode

CAN_BITRATE = 500000

# Map actuator (sim joint) names to CAN node IDs
CAN_ACTUATOR_MAP = {
    "R1_Hip": 1,
    "R1_Femur": 2,
    "R1_Tibia": 3,
}

# Motor rotation direction relative to the sim joint axis. The tibia motor is
# mounted facing the opposite way from the hip and femur motors, so it turns backwards.
CAN_ACTUATOR_DIRECTION = {
    "R1_Hip": 1.0,
    "R1_Femur": 1.0,
    "R1_Tibia": -1.0,
}

DEFAULT_POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_pos.json")
CONNECT_TIMEOUT_SEC = 5.0


def choose_port():
    """List serial ports and let user select one."""
    IGNORE_PORTS = ["/dev/cu.debug-console", "/dev/cu.Bluetooth-Incoming-Port"]
    ports = list(list_ports.comports())
    ports = [p for p in ports if p.device not in IGNORE_PORTS]
    if not ports:
        print("No serial ports found.")
        port = input("Enter serial port path manually: ").strip()
        return port or None

    print("\nSelect your CAN bus port:")
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


def connect_to_can() -> CanMotorController:
    """Open the CAN bus, register all motor nodes, and wait for them to report in."""
    port = choose_port()
    bus = CanMotorController(channel=port, bitrate=CAN_BITRATE)
    for node_id in CAN_ACTUATOR_MAP.values():
        bus.register_node(node_id)
    bus.start()
    for node_id in CAN_ACTUATOR_MAP.values():
        bus.send_clear_errors(node_id)
        time.sleep(0.05)

    # Wait for every node to send a heartbeat (and encoder estimates)
    deadline = time.monotonic() + CONNECT_TIMEOUT_SEC
    while bus.get_stale_nodes():
        if time.monotonic() > deadline:
            bus.stop()
            raise RuntimeError(f"No heartbeat from CAN nodes: {bus.get_stale_nodes()}")
        time.sleep(0.05)
    time.sleep(0.2)
    return bus


def enable_motors(bus: CanMotorController):
    """Put all motors into closed-loop position control."""
    for node_id in CAN_ACTUATOR_MAP.values():
        bus.send_set_control_mode(node_id, ControlMode.POSITION_CONTROL, InputMode.PASSTHROUGH)
        time.sleep(0.05)
        bus.send_set_state(node_id, AxisState.CLOSED_LOOP_CONTROL)
        time.sleep(0.05)


def load_default_positions() -> dict[str, float] | None:
    """Load the saved zero positions, if the file has a value for every actuator."""
    if not os.path.exists(DEFAULT_POS_FILE):
        return None
    try:
        with open(DEFAULT_POS_FILE) as f:
            data = json.load(f)
        return {name: float(data[name]) for name in CAN_ACTUATOR_MAP}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"⚠️  {DEFAULT_POS_FILE} is missing values, recalibrating.")
        return None


def calibrate_default_positions(bus: CanMotorController) -> dict[str, float]:
    """Read the motor positions with the leg held in the alignment tool, and save them."""
    input("\n📐 Put the leg in the alignment tool, then press Enter...")
    # Read via SDO: the cyclic encoder messages report 0 while the motors are idle
    default_pos = {
        name: bus.read_position(node_id) for name, node_id in CAN_ACTUATOR_MAP.items()
    }
    with open(DEFAULT_POS_FILE, "w") as f:
        json.dump(default_pos, f, indent=2)
    print(f"Saved zero positions to {DEFAULT_POS_FILE}:")
    for name, pos in default_pos.items():
        print(f"  {name}: {pos:.4f} rad")
    input("\n🔧 Remove the leg alignment tool, then press Enter to start...")
    return default_pos
