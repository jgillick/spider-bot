"""
ODrive USB connection, motor initialization, cleanup, and error polling.

Wraps the odrive Python library. All attribute paths are for ODrive firmware
compatible with the CAN Simple protocol (GIM6010-8, GIM8108-8, etc.).

Enum values match spider-bot/motor/protocol.py for consistency:
  ControlMode.TORQUE_CONTROL  = 1
  InputMode.PASSTHROUGH       = 1
  AxisState.CLOSED_LOOP_CONTROL = 8
  AxisState.IDLE              = 1
"""

from __future__ import annotations

import time

try:
    import odrive
    from odrive.enums import AxisState, ControlMode, InputMode
except ImportError:  # pragma: no cover — hardware not present in CI
    odrive = None  # type: ignore[assignment]

    class AxisState:  # type: ignore[no-redef]
        IDLE = 1
        CLOSED_LOOP_CONTROL = 8

    class ControlMode:  # type: ignore[no-redef]
        TORQUE_CONTROL = 1

    class InputMode:  # type: ignore[no-redef]
        PASSTHROUGH = 1


# Axis error codes from ODrive firmware (subset covering common faults)
_AXIS_ERROR_NAMES: dict[int, str] = {
    0x00000001: "INVALID_STATE",
    0x00000002: "DC_BUS_UNDER_VOLTAGE",
    0x00000004: "DC_BUS_OVER_VOLTAGE",
    0x00000008: "CURRENT_MEASUREMENT_TIMEOUT",
    0x00000010: "BRAKE_RESISTOR_DISARMED",
    0x00000020: "MOTOR_DISARMED",
    0x00000040: "MOTOR_FAILED",
    0x00000080: "SENSORLESS_ESTIMATOR_FAILED",
    0x00000100: "ENCODER_FAILED",
    0x00000200: "CONTROLLER_FAILED",
    0x00000400: "POS_CTRL_DURING_SENSORLESS",
    0x00000800: "WATCHDOG_TIMER_EXPIRED",
    0x00001000: "MIN_ENDSTOP_PRESSED",
    0x00002000: "MAX_ENDSTOP_PRESSED",
    0x00004000: "ESTOP_REQUESTED",
    0x00010000: "HOMING_WITHOUT_ENDSTOP",
    0x00020000: "OVER_TEMP",
}


def decode_axis_error(error_code: int) -> str:
    if error_code == 0:
        return "NO_ERROR"
    names = [name for mask, name in _AXIS_ERROR_NAMES.items() if error_code & mask]
    return " | ".join(names) if names else f"UNKNOWN_ERROR(0x{error_code:08X})"


def connect(serial: str | None = None, timeout: float = 10.0):
    """
    Connect to an ODrive controller over USB.

    Args:
        serial: Optional serial number string to target a specific device.
        timeout: Seconds to wait before raising ConnectionError.

    Returns:
        The axis0 object (odrv.axis0).
    """
    if odrive is None:
        raise ImportError("odrive package is not installed. Run: pip install odrive")

    kwargs: dict = {"timeout": timeout}
    if serial:
        kwargs["serial_number"] = serial

    try:
        odrv = odrive.find_any(**kwargs)
    except Exception as exc:
        raise ConnectionError(
            f"No ODrive found within {timeout}s. "
            "Check USB connection and that the controller is powered."
        ) from exc

    if odrv is None:
        raise ConnectionError(
            f"No ODrive found within {timeout}s. "
            "Check USB connection and that the controller is powered."
        )

    return odrv.axis0


def initialize(axis, max_torque: float) -> None:
    """
    Initialize the motor into torque control / closed-loop mode.

    Sets control_mode and input_mode, then transitions to CLOSED_LOOP_CONTROL.
    Prints confirmation on success.
    """
    axis.controller.config.control_mode = ControlMode.TORQUE_CONTROL
    axis.controller.config.input_mode = InputMode.PASSTHROUGH
    axis.controller.input_torque = 0.0
    time.sleep(0.05)

    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.2)

    err = poll_errors(axis)
    if err:
        cleanup(axis)
        raise RuntimeError(f"Axis error after initialization: {err}")

    print("Motor initialized: torque control, closed-loop.")
    print(f"Max torque cap: {max_torque:.3f} N·m (motor-side)")


def cleanup(axis) -> None:
    """
    Zero torque and transition the axis to IDLE.

    Designed to be called from finally blocks and signal handlers.
    Swallows exceptions to avoid masking the original error.
    """
    try:
        axis.controller.input_torque = 0.0
        time.sleep(0.05)
        axis.requested_state = AxisState.IDLE
    except Exception:
        pass


def verify_encoder_frame(axis, gear_ratio: float) -> None:
    """
    Interactive encoder frame-of-reference check.

    Prompts the user to rotate the output shaft one full revolution, then
    reports whether pos_estimate is rotor-side or output-shaft-side.
    The motor should be in torque control with input_torque=0 (freewheeling).
    """
    print()
    print("Encoder frame-of-reference check")
    print("─" * 40)
    print("With input_torque=0 the motor is freewheeling — rotate it freely.")
    print()
    print("  1. Note the current output shaft position.")
    print("  2. Rotate the output shaft exactly ONE full revolution (360°).")
    print("  3. Press Enter when done.")
    print()

    pos_before = float(axis.encoder.pos_estimate)

    try:
        input("Press Enter after rotating one full output shaft revolution...")
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    pos_after = float(axis.encoder.pos_estimate)
    delta = abs(pos_after - pos_before)

    print(f"\nΔpos = {delta:.4f} turns (raw encoder units)")
    print()

    if abs(delta - 1.0) < 0.15:
        print("→ OUTPUT-SHAFT: encoder reads output-shaft turns.")
        print("  Velocity is already in joint rad/s after × 2π.")
        print("  sysid fitting does NOT need to divide by gear ratio.")
    elif abs(delta - gear_ratio) < gear_ratio * 0.1:
        print(f"→ ROTOR-SIDE: encoder reads {delta:.2f} turns per output revolution.")
        print(f"  Matches gear ratio {gear_ratio:.1f}. (This is the expected result.)")
        print("  sysid fitting divides vel_estimate by gear_ratio — already applied.")
    else:
        print(f"→ UNEXPECTED: Δpos={delta:.4f} turns.")
        print(f"  Expected ~1.0 (output-shaft) or ~{gear_ratio:.1f} (rotor-side).")
        print("  Check that exactly one full output revolution was performed and rerun.")

    print()


def poll_errors(axis) -> str | None:
    """
    Read axis.error.

    Returns a human-readable error name string, or None if no error.
    """
    try:
        code = int(axis.error)
    except Exception:
        return "COMMUNICATION_ERROR"
    if code == 0:
        return None
    return decode_axis_error(code)
