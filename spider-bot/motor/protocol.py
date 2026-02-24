"""
ODrive CAN Simple Protocol constants and enums.

Shared definitions used by both the blocking (simple_can) and non-blocking
(can_motor_controller) controller interfaces.
"""

import math
from enum import IntEnum

TWO_PI = 2.0 * math.pi

# Motor position scale: 0–8 = one full revolution (8:1 reduction naming).
# 1 motor unit = π/4 rad = 45°
MOTOR_UNITS_PER_REV = 8.0
MOTOR_TO_RAD = TWO_PI / MOTOR_UNITS_PER_REV  # π/4
RAD_TO_MOTOR = MOTOR_UNITS_PER_REV / TWO_PI  # 4/π

# MIT protocol ranges (output-shaft side, per SteadyWin manual)
MIT_POS_MIN, MIT_POS_MAX = -12.5, 12.5  # rad
MIT_VEL_MIN, MIT_VEL_MAX = -65.0, 65.0  # rad/s
MIT_KP_MIN, MIT_KP_MAX = 0.0, 500.0
MIT_KD_MIN, MIT_KD_MAX = 0.0, 5.0
MIT_TORQUE_MIN, MIT_TORQUE_MAX = -50.0, 50.0  # Nm

# How long (seconds) without a heartbeat before is_connected becomes False.
# Default heartbeat_rate_ms on the motor is 100 ms, so 0.5 s is generous.
HEARTBEAT_TIMEOUT = 0.5

# SDO endpoint IDs
ENCODER_OFFSET_ENDPOINT_ID = 362
ENCODER_POSITION_ESTIMATE_ID = 349


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
