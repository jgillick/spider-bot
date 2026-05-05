"""
Motor sysid CLI entry point.

Usage:
    python -m hardware.sysid --gear-ratio 8

Or, if installed via pyproject.toml scripts:
    sysid --gear-ratio 8
"""

from __future__ import annotations

import argparse
import signal
import sys

from . import experiments as exp
from . import fitting, motor, output
from .experiments import SafetyLimitError

_WARMUP_MSG = """
┌─────────────────────────────────────────────────────────────┐
│  WARM-UP REMINDER                                           │
│  Run the motor for 5-10 minutes at operating temperature   │
│  before collecting sysid data.                              │
└─────────────────────────────────────────────────────────────┘
"""

_MENU = """
Sysid Menu:
  1. Verify encoder frame of reference  ← run first
  2. Rotor inertia (J) + Viscous damping (b)
  3. Coulomb friction (τ_c)
  4. Static friction (τ_static)
  5. kp/kd step response
  6. Quit
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sysid",
        description="Motor system identification CLI for ODrive-compatible controllers.",
    )
    p.add_argument(
        "--serial",
        default=None,
        help="ODrive serial number (optional; first found if omitted)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="USB connection timeout in seconds (default: 10)",
    )
    p.add_argument(
        "--kt",
        type=float,
        default=None,
        help="Override firmware torque constant Kt (N·m/A)",
    )
    p.add_argument(
        "--gear-ratio",
        type=float,
        required=True,
        metavar="RATIO",
        help="Gear ratio: rotor turns per output shaft turn (e.g. 8 for 8:1)",
    )
    p.add_argument(
        "--max-torque",
        type=float,
        default=1.0,
        metavar="NM",
        help="Maximum motor-side torque in N·m (default: 1.0)",
    )
    p.add_argument(
        "--trials", type=int, default=5, help="Trials per experiment (default: 5)"
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.01,
        metavar="SEC",
        help="Encoder polling interval in seconds (default: 0.01 = 100 Hz)",
    )
    p.add_argument(
        "--kp-test",
        type=float,
        default=5.0,
        help=(
            "kp for kp/kd step-response experiment (default: 5). "
            "The Python control loop runs at ~100 Hz; gains stable in simulation "
            "at 1 kHz may oscillate here. Start low and increase if the fit is poor."
        ),
    )
    p.add_argument(
        "--kd-test",
        type=float,
        default=0.2,
        help="kd for kp/kd step-response experiment (default: 0.2)",
    )
    return p


def _run_inertia(axis, args) -> None:
    print("\nRunning: Rotor inertia (J) + Viscous damping (b)")
    trials = exp.run_inertia_experiment(
        axis,
        args.gear_ratio,
        args.max_torque,
        trials=args.trials,
        poll_interval=args.poll_interval,
    )
    J, b, J_spread, b_spread = fitting.fit_inertia_damping(
        trials, tau_step=args.max_torque * 0.5
    )
    print(output.format_inertia_result(J, b, J_spread, b_spread))
    return J, b


def _run_coulomb(axis, args) -> float:
    print("\nRunning: Coulomb friction (τ_c)")
    trials = exp.run_coulomb_experiment(
        axis,
        args.gear_ratio,
        args.max_torque,
        trials=args.trials,
        poll_interval=args.poll_interval,
    )
    tau_c_cw, tau_c_ccw, spread_cw, spread_ccw = fitting.fit_coulomb(trials)
    print(output.format_coulomb_result(tau_c_cw, tau_c_ccw, spread_cw, spread_ccw))
    return (tau_c_cw + tau_c_ccw) / 2.0


def _run_static(axis, args) -> None:
    print("\nRunning: Static friction (τ_static)")
    trials = exp.run_static_experiment(
        axis,
        args.gear_ratio,
        args.max_torque,
        trials=args.trials,
        poll_interval=args.poll_interval,
    )
    tau_s_cw, tau_s_ccw = fitting.fit_static(trials)
    print(output.format_static_result(tau_s_cw, tau_s_ccw))


def _run_kp_kd(axis, args, J: float | None, b: float | None) -> None:
    if J is None or b is None:
        print(
            "\nCannot run kp/kd experiment: rotor inertia (J) and damping (b) not yet measured."
        )
        print("Run experiment 2 (Rotor inertia + Viscous damping) first.")
        print(
            "kp/kd fitting uses J and b as fixed inputs — without real values the"
            " MIT loop will be unstable and will exceed velocity safety limits."
        )
        return

    step_size = 0.2
    print("\nRunning: kp/kd step response (soft MIT impedance loop, ramped target)")

    ramp_time: float | None = None  # auto on first attempt
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            trials, ramp_time_used = exp.run_kp_kd_experiment(
                axis,
                args.gear_ratio,
                args.max_torque,
                kp_test=args.kp_test,
                kd_test=args.kd_test,
                trials=args.trials,
                poll_interval=args.poll_interval,
                step_size_rad=step_size,
                ramp_time=ramp_time,
            )
            break
        except SafetyLimitError as exc:
            if attempt == max_retries:
                print(
                    f"\n  Still hitting velocity limit after {max_retries} ramp doublings.",
                    file=sys.stderr,
                )
                print(
                    f"  Current kp={args.kp_test}, kd={args.kd_test}. "
                    "High kp causes oscillation in a 100 Hz Python loop even with a slow ramp.",
                    file=sys.stderr,
                )
                print(
                    "  Retry with lower gains, e.g.:  --kp-test 2 --kd-test 0.1",
                    file=sys.stderr,
                )
                raise
            ramp_time = exc.ramp_time * 2.0
            print(
                f"  Velocity limit hit. Retrying with longer ramp ({ramp_time:.2f}s)..."
            )

    kp, kd = fitting.fit_kp_kd(
        trials,
        p_des=step_size,
        J=J,
        b=b,
        kp_test=args.kp_test,
        kd_test=args.kd_test,
        ramp_time=ramp_time_used,
    )
    print(output.format_kp_kd_result(kp, kd))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(_WARMUP_MSG)
    print(
        f"Gear ratio: {args.gear_ratio:.1f}  ({args.gear_ratio:.1f} rotor turns per 1 output shaft turn)"
    )
    print(f"Max torque: {args.max_torque:.3f} N·m (motor-side)")

    # Connect
    print(f"\nConnecting to ODrive (timeout {args.timeout}s)...")
    try:
        axis = motor.connect(serial=args.serial, timeout=args.timeout)
    except (ConnectionError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Read / override Kt
    try:
        kt_firmware = float(axis.config.motor.torque_constant)
    except Exception:
        kt_firmware = None

    if args.kt is not None:
        kt = args.kt
        print(
            f"Kt: {kt:.4f} N·m/A  (overridden via --kt; firmware value: {kt_firmware})"
        )
    elif kt_firmware is not None:
        kt = kt_firmware
        print(f"Kt: {kt:.4f} N·m/A  (read from firmware)")
    else:
        print(
            "WARNING: could not read Kt from firmware and --kt not supplied. Torque calibration uncertain."
        )
        kt = None

    # Register cleanup handler for Ctrl-C
    def _sigint_handler(sig, frame):
        print("\n\nInterrupted — zeroing torque and idling motor...")
        motor.cleanup(axis)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Initialize motor
    try:
        motor.initialize(axis, args.max_torque)
    except Exception as exc:
        print(f"ERROR during motor initialization: {exc}", file=sys.stderr)
        motor.cleanup(axis)
        sys.exit(1)

    # Session state: carry J and b for kp/kd experiment
    J_est: float | None = None
    b_est: float | None = None

    # Menu loop
    while True:
        print(_MENU)
        try:
            choice = input("Select [1-5]: ").strip()
        except EOFError:
            break

        if choice == "1":
            motor.verify_encoder_frame(axis, args.gear_ratio)

        elif choice == "2":
            try:
                J_est, b_est = _run_inertia(axis, args)
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "3":
            try:
                _run_coulomb(axis, args)
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "4":
            try:
                _run_static(axis, args)
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "5":
            try:
                _run_kp_kd(axis, args, J_est, b_est)
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "6" or choice.lower() in ("q", "quit", "exit"):
            break

        else:
            print("Invalid choice. Enter 1–6.")

    print("\nSession complete. Zeroing torque and idling motor.")
    motor.cleanup(axis)


if __name__ == "__main__":
    main()
