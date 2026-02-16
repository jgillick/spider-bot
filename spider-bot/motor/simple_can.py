"""
ODrive CAN Simple Protocol API

A Python API for controlling ODrive motors via CAN bus using the CAN Simple protocol.
SteadyWin GIM6010-8: position is in motor units 0 - 8 per revolution (0 and 8 = same, 4 = 180°).
API exposes position/velocity in radians; conversion is applied internally.
"""

import can
import math
import struct
import time
from enum import IntEnum
from typing import Optional

TWO_PI = 2.0 * math.pi

# Motor position scale: 0–8 = one full revolution (8:1 reduction naming).
# 1 motor unit = π/4 rad = 45°
MOTOR_UNITS_PER_REV = 8.0
MOTOR_TO_RAD = TWO_PI / MOTOR_UNITS_PER_REV  # π/4
RAD_TO_MOTOR = MOTOR_UNITS_PER_REV / TWO_PI  # 4/π

# SDO endpoint IDs
ENCODER_OFFSET_ENDPOINT_ID = 362
ENCODER_POSITION_ESTIMATE_ID = 349


def rad_to_rev(rad: float) -> float:
    """Convert radians to revolutions (for vel_ff / protocol fields that use rev/s)."""
    return rad / TWO_PI


class AxisState(IntEnum):
    """ODrive axis states"""

    UNDEFINED = 0
    IDLE = 1
    STARTUP_SEQUENCE = 2
    FULL_CALIBRATION_SEQUENCE = 3
    MOTOR_CALIBRATION = 4
    ENCODER_INDEX_SEARCH = 6
    ENCODER_OFFSET_CALIBRATION = 7
    CLOSED_LOOP_CONTROL = 8
    LOCKIN_SPIN = 9
    ENCODER_DIR_FIND = 10
    HOMING = 11
    ENCODER_HALL_POLARITY_CALIBRATION = 12
    ENCODER_HALL_PHASE_CALIBRATION = 13


class ControlMode(IntEnum):
    """ODrive control modes"""

    VOLTAGE_CONTROL = 0
    TORQUE_CONTROL = 1
    VELOCITY_CONTROL = 2
    POSITION_CONTROL = 3


class InputMode(IntEnum):
    """ODrive input modes"""

    INACTIVE = 0
    PASSTHROUGH = 1
    VEL_RAMP = 2
    POS_FILTER = 3
    MIX_CHANNELS = 4
    TRAP_TRAJ = 5
    TORQUE_RAMP = 6
    MIRROR = 7
    TUNING = 8


class CANCommand(IntEnum):
    """CAN Simple protocol command IDs"""

    HEARTBEAT = 0x001
    ESTOP = 0x002
    GET_ERROR = 0x003
    RXSDO = 0x004
    TXSDO = 0x005
    SET_AXIS_NODE_ID = 0x006
    SET_AXIS_STATE = 0x007
    MIT_CONTROL = 0x008
    GET_ENCODER_ESTIMATES = 0x009
    GET_ENCODER_COUNT = 0x00A
    SET_CONTROLLER_MODE = 0x00B
    SET_INPUT_POS = 0x00C
    SET_INPUT_VEL = 0x00D
    SET_INPUT_TORQUE = 0x00E
    SET_LIMITS = 0x00F
    START_ANTICOGGING = 0x010
    SET_TRAJ_VEL_LIMIT = 0x011
    SET_TRAJ_ACCEL_LIMITS = 0x012
    SET_TRAJ_INERTIA = 0x013
    GET_IQ = 0x014
    GET_TEMPERATURE = 0x015
    REBOOT = 0x016
    GET_BUS_VOLTAGE_CURRENT = 0x017
    CLEAR_ERRORS = 0x018
    SET_LINEAR_COUNT = 0x019
    SET_POS_GAIN = 0x01A
    SET_VEL_GAINS = 0x01B
    GET_TORQUES = 0x01C
    GET_POWERS = 0x01D
    SAVE_CONFIGURATION = 0x01E
    DISABLE_CAN = 0x01F


def make_can_id(cmd_id: int, node_id: int) -> int:
    """Create CAN arbitration ID from command and node ID"""
    return (node_id << 5) | cmd_id


class SimpleCanController:
    """
    Controller for the Simple CAN protocol (specific to the SteadyWin GIM6010-8 motor).
    """

    def __init__(
        self,
        node_id: int,
        port: str = "/dev/cu.usbmodem1101",
        bitrate: int = 500000,
        interface: str = "slcan",
    ):
        """
        Initialize controller.

        Args:
            node_id: CAN node ID of the ODrive axis
            port: Serial port for CAN adapter
            bitrate: CAN bus bitrate (default 500kbps)
            interface: python-can interface type (default "slcan")
        """
        self.node_id = node_id
        self.port = port
        self.bitrate = bitrate
        self.interface = interface

        # State
        self._position = 0.0
        self._velocity = 0.0
        self._axis_state = AxisState.UNDEFINED
        self._axis_error = 0
        self._target_position: Optional[float] = None
        self._zero_offset = 0.0  # Virtual zero (motor units)

        self._bus: Optional[can.Bus] = None

    def connect(self) -> None:
        """Open CAN bus connection"""
        if self._bus is not None:
            return

        self._bus = can.interface.Bus(
            channel=self.port,
            interface=self.interface,
            bitrate=self.bitrate,
        )

    def disconnect(self) -> None:
        """Close CAN bus connection"""
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    @property
    def is_connected(self) -> bool:
        return self._bus is not None

    @property
    def position(self) -> float:
        """Current position relative to virtual zero (radians)"""
        return (self._position - self._zero_offset) * MOTOR_TO_RAD

    @property
    def raw_position(self) -> float:
        """Raw encoder position (radians)"""
        return self._position * MOTOR_TO_RAD

    @property
    def velocity(self) -> float:
        """Current velocity (rad/s)"""
        return self._velocity * MOTOR_TO_RAD

    @property
    def axis_state(self) -> AxisState:
        """Current axis state"""
        return self._axis_state

    @property
    def axis_error(self) -> int:
        """Current axis error code"""
        return self._axis_error

    @property
    def target_position(self) -> Optional[float]:
        """Target position relative to virtual zero (radians)"""
        if self._target_position is None:
            return None
        return self._target_position * MOTOR_TO_RAD

    @property
    def zero_offset(self) -> float:
        """Current virtual zero offset (radians)"""
        return self._zero_offset * MOTOR_TO_RAD

    def update(self) -> None:
        """Process incoming CAN messages and update state"""
        if self._bus is None:
            raise RuntimeError("Not connected")

        while True:
            msg = self._bus.recv(timeout=0.001)
            if msg is None:
                break

            cmd_id, recv_node = self._parse_id(msg)
            if recv_node != self.node_id:
                continue

            if cmd_id == CANCommand.HEARTBEAT and len(msg.data) >= 5:
                self._axis_error = struct.unpack("<I", msg.data[0:4])[0]
                self._axis_state = AxisState(msg.data[4])

            elif cmd_id == CANCommand.GET_ENCODER_ESTIMATES and len(msg.data) >= 8:
                self._position, self._velocity = struct.unpack("<ff", msg.data)

    def _send(self, cmd_id: int, data: bytes = b"") -> None:
        """Send a CAN command"""
        if self._bus is None:
            raise RuntimeError("Not connected")

        msg = can.Message(
            arbitration_id=make_can_id(cmd_id, self.node_id),
            data=data,
            is_extended_id=False,
        )
        self._bus.send(msg)

    def _parse_id(self, msg: can.Message) -> tuple[int, int]:
        """
        Parse the CAN arbitration ID into command ID and node ID

        Args:
            msg: The CAN message to parse

        Returns:
            A tuple containing the command ID and node ID
        """
        can_id = msg.arbitration_id
        cmd_id = can_id & 0x1F
        recv_node = can_id >> 5
        return cmd_id, recv_node

    # === High-level commands ===

    def set_zero_here(self) -> None:
        """Set current position as the new virtual zero"""
        self._zero_offset = self._position
        if self._target_position is not None:
            self._target_position = 0.0

    def reset_zero(self) -> None:
        """Reset virtual zero to encoder zero"""
        self._zero_offset = 0.0

    def rxsdo_write(self, endpoint_id: int, value: bytes) -> None:
        """
        Write 4 bytes to a parameter via RxSdo (CAN Simple cmd 0x004).
        value must be 4 bytes (e.g. struct.pack('<f', x) or struct.pack('<I', x)).
        """
        if len(value) != 4:
            raise ValueError("RxSdo value must be 4 bytes")
        # Format: opcode(1)=write, endpoint_id(2) LE, reserved(1)=0, value(4)
        data = struct.pack("<BHB", 1, endpoint_id & 0xFFFF, 0) + value
        self._send(CANCommand.RXSDO, data)

    def rxsdo_read_float(self, endpoint_id: int, timeout_s: float = 1.5) -> float:
        """
        Read a float from an endpoint via RxSdo read (opcode 0); wait for TxSdo response.
        Raises TimeoutError if the device does not respond within timeout_s.
        """
        if self._bus is None:
            raise RuntimeError("Not connected")
        # Drain pending messages so we don't miss the response
        drained = 0
        while self._bus.recv(timeout=0) is not None:
            drained += 1
        # Opcode 0 = read; value bytes ignored for read request
        data = struct.pack("<BHB", 0, endpoint_id & 0xFFFF, 0) + b"\x00\x00\x00\x00"
        self._send(CANCommand.RXSDO, data)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self._bus.recv(timeout=0.05)
            if msg is None:
                continue

            cmd_id, recv_node = self._parse_id(msg)
            if recv_node != self.node_id:
                continue
            if cmd_id not in (CANCommand.TXSDO, CANCommand.RXSDO):
                continue
            if len(msg.data) < 8:
                continue

            # TxSDO/RxSDO response: byte0 reserved, bytes 1-2 endpoint_id LE, byte3 reserved, bytes 4-7 value
            response_endpoint = struct.unpack("<H", msg.data[1:3])[0]
            if response_endpoint != (endpoint_id & 0xFFFF):
                continue
            value = struct.unpack("<f", msg.data[4:8])[0]
            return value
        raise TimeoutError(
            f"rxsdo_read_float endpoint {endpoint_id}: no response within {timeout_s}s"
        )

    def rxsdo_write_float(self, endpoint_id: int, value: float) -> None:
        """Write a float to an endpoint (e.g. encoder offset)."""
        self.rxsdo_write(endpoint_id, struct.pack("<f", value))

    def save_zero_to_motor(self) -> None:
        """
        Set current position as zero and persist to the motor's index_offset value.
        """
        # Use position from normal CAN feedback (motor units 0–8), not SDO read of endpoint 336
        current_offset = self.rxsdo_read_float(ENCODER_OFFSET_ENDPOINT_ID)
        pos_estimate = self.rxsdo_read_float(ENCODER_POSITION_ESTIMATE_ID)
        new_offset = (pos_estimate + current_offset) % MOTOR_UNITS_PER_REV
        self.rxsdo_write_float(ENCODER_OFFSET_ENDPOINT_ID, new_offset)
        time.sleep(0.05)
        self.save_configuration()

    def move_to(
        self, position: float, vel_ff: float = 0.0, torque_ff: float = 0.0
    ) -> None:
        """
        Move to position.
        If a virtual zero is set, the position is relative to that.

        Args:
            position: Target position in radians
            vel_ff: Velocity feedforward (rad/s)
            torque_ff: Torque feedforward (Nm)
        """
        motor_pos = position * RAD_TO_MOTOR
        self._target_position = motor_pos
        absolute_motor = motor_pos + self._zero_offset
        vel_ff_units = int(rad_to_rev(vel_ff) / 0.001)
        data = struct.pack("<fhh", absolute_motor, vel_ff_units, int(torque_ff / 0.001))
        self._send(CANCommand.SET_INPUT_POS, data)

    def move_by(
        self, delta: float, vel_ff: float = 0.0, torque_ff: float = 0.0
    ) -> None:
        """Move by a relative amount from current position."""
        new_pos = self.position + delta
        self.move_to(new_pos, vel_ff, torque_ff)

    def set_velocity(self, velocity: float, torque_ff: float = 0.0) -> None:
        """Set velocity (requires velocity control mode)."""
        vel_rev = rad_to_rev(velocity)
        data = struct.pack("<ff", vel_rev, torque_ff)
        self._send(CANCommand.SET_INPUT_VEL, data)

    def set_torque(self, torque: float) -> None:
        """Set torque (requires torque control mode)."""
        data = struct.pack("<f", torque)
        self._send(CANCommand.SET_INPUT_TORQUE, data)

    def enable(self) -> None:
        """
        Enable closed loop control (position mode with passthrough).
        """
        self.clear_errors()
        time.sleep(0.05)
        self.set_control_mode(ControlMode.POSITION_CONTROL, InputMode.PASSTHROUGH)
        time.sleep(0.05)
        self.set_state(AxisState.CLOSED_LOOP_CONTROL)
        time.sleep(0.05)

    def disable(self) -> None:
        """Disable motor (set to idle)"""
        self.set_state(AxisState.IDLE)

    def estop(self) -> None:
        """Emergency stop"""
        self._send(CANCommand.ESTOP)

    # === Low-level commands ===

    def set_state(self, state: AxisState) -> None:
        """Set axis state"""
        data = struct.pack("<I", int(state))
        self._send(CANCommand.SET_AXIS_STATE, data)

    def set_control_mode(
        self, control_mode: ControlMode, input_mode: InputMode
    ) -> None:
        """Set controller mode and input mode"""
        data = struct.pack("<II", int(control_mode), int(input_mode))
        self._send(CANCommand.SET_CONTROLLER_MODE, data)

    def clear_errors(self) -> None:
        """Clear axis errors"""
        self._send(CANCommand.CLEAR_ERRORS, struct.pack("<B", 0))

    def save_configuration(self) -> None:
        """Save configuration to flash (motor will reboot)."""
        self._send(CANCommand.SAVE_CONFIGURATION)

    def reboot(self) -> None:
        """Reboot the motor (clears runtime state, does not erase saved config)."""
        self._send(CANCommand.REBOOT, struct.pack("<B", 0) + b"\x00" * 7)
