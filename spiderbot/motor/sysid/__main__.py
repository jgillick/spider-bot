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

from . import motor
from .coulomb import fit_coulomb, format_coulomb_result, run_coulomb_experiment
from .inertia import fit_inertia_damping, format_inertia_result, run_inertia_experiment
from .static import fit_static, format_static_result, run_static_experiment
from .utils import SafetyLimitError

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
  5. Quit
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
        "--consecutive-required",
        type=int,
        default=15,
        metavar="N",
        help=(
            "Consecutive samples above the motion threshold required to declare "
            "breakaway in ramp experiments (default: 15 ≈ 150 ms at 100 Hz). "
            "Long enough to reject cogging events on multi-pole BLDC motors. "
            "Decrease if motion onset is genuinely brief."
        ),
    )
    return p


_ANTICOGGING_WARNING = """
┌─────────────────────────────────────────────────────────────┐
│  WARNING: ANTICOGGING IS ENABLED                            │
│  axis.controller.config.anticogging.anticogging_enabled     │
│  is True. The anticogging feedforward adds a position-      │
│  dependent torque on top of your commanded torque, making   │
│  Coulomb and static friction measurements unreliable.       │
│                                                             │
│  Disable it before running friction experiments:            │
│    odrv0.axis0.controller.config.anticogging               │
│        .anticogging_enabled = False                         │
│    odrv0.save_configuration()                               │
└─────────────────────────────────────────────────────────────┘
"""


def _warn_if_anticogging_enabled(axis) -> None:
    """Print a warning if anticogging is enabled on the axis.

    Silently does nothing if the firmware version does not expose the
    anticogging config path — older ODrive firmware used a different layout.
    """
    try:
        enabled = axis.controller.config.anticogging.anticogging_enabled
    except AttributeError:
        return  # firmware does not expose this path — skip
    if enabled:
        print(_ANTICOGGING_WARNING, file=sys.stderr)


def _reinitialize_or_exit(axis, args, parser) -> None:
    """After a SafetyLimitError the motor is in IDLE state.

    Try to re-arm it so the menu remains usable. If re-initialization
    fails, exit immediately rather than letting the user run experiments
    against an un-energized axis (which would silently produce garbage data).
    """
    print("Attempting to re-initialize motor for continued use...", file=sys.stderr)
    try:
        motor.initialize(axis, args.max_torque)
        print("Motor re-initialized. You may continue.", file=sys.stderr)
    except Exception as reinit_exc:
        print(
            f"Re-initialization failed: {reinit_exc}\n"
            "Exiting to avoid running experiments on an un-energized axis.",
            file=sys.stderr,
        )
        motor.cleanup(axis)
        sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Input validation ---
    if args.gear_ratio <= 0:
        parser.error(f"--gear-ratio must be > 0, got {args.gear_ratio}")
    if args.max_torque <= 0:
        parser.error(f"--max-torque must be > 0, got {args.max_torque}")
    if args.trials < 1:
        parser.error(f"--trials must be >= 1, got {args.trials}")

    print(_WARMUP_MSG)
    print(
        f"Gear ratio: {args.gear_ratio:.1f}  "
        f"({args.gear_ratio:.1f} rotor turns per 1 output shaft turn)"
    )
    print(f"Max torque: {args.max_torque:.3f} N·m (motor-side)")

    # Connect
    print(f"\nConnecting to ODrive (timeout {args.timeout}s)...")
    try:
        axis = motor.connect(serial=args.serial, timeout=args.timeout)
    except (ConnectionError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Register cleanup handler for Ctrl-C (before motor.initialize so it is
    # always active once we have a live axis handle).
    def _sigint_handler(sig, frame):
        print("\n\nInterrupted — zeroing torque and idling motor...")
        _zeroed = False
        try:
            axis.controller.input_torque = 0.0
            _zeroed = True
        except Exception:
            pass
        try:
            motor.cleanup(axis)
        except Exception:
            pass
        if not _zeroed:
            print(
                "WARNING: torque zero command FAILED — hardware may still be live. "
                "Power off the controller immediately.",
                file=sys.stderr,
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Initialize motor
    try:
        motor.initialize(axis, args.max_torque)
    except Exception as exc:
        print(f"ERROR during motor initialization: {exc}", file=sys.stderr)
        motor.cleanup(axis)
        sys.exit(1)

    _warn_if_anticogging_enabled(axis)

    # Menu loop
    while True:
        print(_MENU)
        try:
            choice = input("Select [1-5]: ").strip()
        except EOFError:
            break

        if choice == "1":
            try:
                motor.verify_encoder_frame(axis, args.gear_ratio)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "2":
            try:
                print("\nRunning: Rotor inertia (J) + Viscous damping (b)")
                step_torque = min(0.5, args.max_torque * 0.5)
                trials = run_inertia_experiment(
                    axis,
                    args.gear_ratio,
                    args.max_torque,
                    trials=args.trials,
                    poll_interval=args.poll_interval,
                    step_torque=step_torque,
                )
                J, b, J_spread, b_spread = fit_inertia_damping(
                    trials, tau_step=step_torque
                )
                print(format_inertia_result(J, b, J_spread, b_spread))
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
                _reinitialize_or_exit(axis, args, parser)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "3":
            try:
                print("\nRunning: Coulomb friction (τ_c)")
                trials = run_coulomb_experiment(
                    axis,
                    args.gear_ratio,
                    args.max_torque,
                    trials=args.trials,
                    poll_interval=args.poll_interval,
                    consecutive_required=args.consecutive_required,
                )
                tau_c_cw, tau_c_ccw, spread_cw, spread_ccw = fit_coulomb(trials)
                print(format_coulomb_result(tau_c_cw, tau_c_ccw, spread_cw, spread_ccw))
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
                _reinitialize_or_exit(axis, args, parser)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "4":
            try:
                print("\nRunning: Static friction (τ_static)")
                trials = run_static_experiment(
                    axis,
                    args.gear_ratio,
                    args.max_torque,
                    trials=args.trials,
                    poll_interval=args.poll_interval,
                    consecutive_required=args.consecutive_required,
                )
                tau_s_cw, tau_s_ccw = fit_static(trials)
                print(format_static_result(tau_s_cw, tau_s_ccw))
            except SafetyLimitError as exc:
                print(f"\nSafety limit reached: {exc}", file=sys.stderr)
                _reinitialize_or_exit(axis, args, parser)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                motor.cleanup(axis)
                sys.exit(1)

        elif choice == "5" or choice.lower() in ("q", "quit", "exit"):
            break

        else:
            print("Invalid choice. Enter 1-5.")

    print("\nSession complete. Zeroing torque and idling motor.")
    motor.cleanup(axis)


if __name__ == "__main__":
    main()
