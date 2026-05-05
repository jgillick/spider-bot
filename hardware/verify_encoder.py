"""
Standalone encoder frame-of-reference verification script.

Confirms whether axis0.encoder.pos_estimate is rotor-side or output-shaft-side
before running sysid experiments. Run once per motor/controller combination.

Usage:
    python hardware/verify_encoder.py --gear-ratio 8
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running as a script from the repo root or hardware/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.sysid import motor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ODrive encoder frame of reference (rotor vs output shaft)."
    )
    parser.add_argument("--gear-ratio", type=float, required=True, metavar="RATIO",
                        help="Expected gear ratio (rotor turns per output shaft turn, e.g. 8)")
    parser.add_argument("--serial", default=None, help="ODrive serial number (optional)")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Connecting to ODrive (timeout {args.timeout}s)...")
    try:
        axis = motor.connect(serial=args.serial, timeout=args.timeout)
    except (ConnectionError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Connected.")
    motor.verify_encoder_frame(axis, args.gear_ratio)


if __name__ == "__main__":
    main()
