"""
Shared types, safety helpers, and output utilities for sysid experiments.

Contains:
  - SafetyLimitError          recoverable safety exception
  - StepTrial / RampTrial     experiment result dataclasses
  - Hardware helpers           _vel_to_rad_s, _safe_torque, _check_errors, _poll_loop
  - Ramp loop                  _run_ramp_experiment (shared by coulomb and static)
  - Fitting helpers            _deduplicate, _velocity_model
  - Output helpers             _header, _noisy_value, _dr_range

Nothing here calls real hardware directly — all ODrive I/O is injected via axis objects.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .motor import cleanup, poll_errors


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class SafetyLimitError(RuntimeError):
    """Raised when a soft safety limit is exceeded during an experiment.

    Recoverable — the motor is already zeroed before this is raised.
    """


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWO_PI = 2.0 * math.pi

# Default motion-onset threshold for Coulomb / static ramp experiments (rad/s)
_MOTION_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class StepTrial:
    """Raw data from a torque-step experiment."""

    t: np.ndarray  # seconds from step onset
    vel: np.ndarray  # output-shaft rad/s (gear-ratio corrected)
    iq: np.ndarray  # Iq_measured A (sanity check only)


@dataclass(eq=False)
class RampTrial:
    """Result from a torque-ramp (Coulomb / static) experiment."""

    breakaway_torque: float  # motor-side N·m at which motion onset detected
    direction: int  # +1 (CW) or -1 (CCW)


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _vel_to_rad_s(vel_estimate: float, gear_ratio: float) -> float:
    """Convert rotor-side encoder velocity (turns/s) to output-shaft rad/s."""
    if gear_ratio == 0:
        raise ValueError("gear_ratio must be non-zero")
    return vel_estimate * TWO_PI / gear_ratio


def _safe_torque(axis, torque: float, max_torque: float) -> None:
    """Send a torque command, raising SafetyLimitError if it would exceed the cap.

    Zeros the torque before raising so the motor is safe regardless of what
    the caller does next.
    """
    if abs(torque) > max_torque:
        try:
            axis.controller.input_torque = 0.0
        except Exception:
            pass
        raise SafetyLimitError(
            f"Torque command {torque:.4f} N·m exceeds --max-torque {max_torque:.4f} N·m. "
            "Raise --max-torque or reduce the experiment step size."
        )
    axis.controller.input_torque = torque


def _check_errors(axis) -> None:
    """Poll all ODrive error registers; raise RuntimeError on any fault."""
    err = poll_errors(axis)
    if err:
        cleanup(axis)
        raise RuntimeError(f"Axis error during experiment: {err}")


def _poll_loop(
    axis,
    gear_ratio: float,
    duration: float,
    poll_interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poll encoder at poll_interval for duration seconds. Returns (t, vel, iq).

    Checks axis errors every ~10 samples. Zeros torque and raises RuntimeError
    on USB loss or axis fault detected mid-loop.
    """
    t_list: list[float] = []
    vel_list: list[float] = []
    iq_list: list[float] = []

    t_start = time.monotonic()
    sample_count = 0
    while True:
        t_now = time.monotonic() - t_start
        if t_now >= duration:
            break

        # Encoder read — guard against USB drop
        try:
            vel = _vel_to_rad_s(axis.encoder.vel_estimate, gear_ratio)
        except Exception as exc:
            try:
                axis.controller.input_torque = 0.0
            except Exception:
                pass
            raise RuntimeError("Lost contact with motor during polling loop") from exc

        vel_list.append(vel)
        t_list.append(t_now)

        try:
            iq_list.append(float(axis.motor.current_control.Iq_measured))
        except (AttributeError, TypeError, ValueError):
            iq_list.append(float("nan"))

        sample_count += 1
        # Check axis error registers every ~10 samples (~100 ms at default poll rate)
        if sample_count % 10 == 0:
            _check_errors(axis)

        time.sleep(poll_interval)

    return (np.array(t_list), np.array(vel_list), np.array(iq_list))


def _run_ramp_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int,
    poll_interval: float,
    ramp_rate: float,
    motion_threshold: float,
    trial_label: str,
    consecutive_required: int = 15,
) -> list[RampTrial]:
    """Inner ramp loop shared by the Coulomb and static experiments.

    Slowly increases torque and records the torque at which velocity first
    exceeds motion_threshold for each direction.

    ``consecutive_required`` controls how many consecutive samples above
    ``motion_threshold`` are needed before breakaway is declared.  The default
    of 15 (≈150 ms at 100 Hz) is long enough to reject cogging events, which
    can sustain velocity above the threshold for 30–100 ms on multi-pole BLDC
    motors without any net rotor displacement.  The torque recorded is the value
    at the *first* sample of the consecutive run, so it is not biased upward by
    the extra samples.
    """
    results: list[RampTrial] = []

    for direction in (1, -1):
        for trial_idx in range(trials):
            print(
                f"  {trial_label} trial {trial_idx + 1}/{trials} "
                f"{'CW' if direction == 1 else 'CCW'}...",
                end=" ",
                flush=True,
            )
            _check_errors(axis)

            ramp_step = 0
            breakaway: float | None = None
            consecutive_above = 0
            breakaway_candidate: float | None = None

            # Compute torque from step count to avoid float accumulation drifting
            # past max_torque and triggering a spurious SafetyLimitError.
            # max(1, ...) ensures _safe_torque is called at least once so an
            # oversized ramp_step_size (ramp_rate * poll_interval > max_torque)
            # still raises SafetyLimitError rather than silently doing nothing.
            ramp_step_size = ramp_rate * poll_interval
            max_steps = max(1, int(max_torque / ramp_step_size))

            while ramp_step < max_steps:
                tau = direction * (ramp_step + 1) * ramp_step_size
                _safe_torque(axis, tau, max_torque)
                time.sleep(poll_interval)

                # Encoder read — guard against USB drop
                try:
                    vel = _vel_to_rad_s(axis.encoder.vel_estimate, gear_ratio)
                except Exception as exc:
                    try:
                        axis.controller.input_torque = 0.0
                    except Exception:
                        pass
                    raise RuntimeError("Lost contact with motor during ramp") from exc

                if abs(vel) >= motion_threshold:
                    if consecutive_above == 0:
                        # Record torque at the *first* sample of the run so the
                        # reported breakaway is not biased upward by the extra
                        # confirmation samples.
                        breakaway_candidate = abs(tau)
                    consecutive_above += 1
                    if consecutive_above >= consecutive_required:
                        breakaway = breakaway_candidate
                        break
                else:
                    # Noise spike — reset the counter and discard the candidate.
                    consecutive_above = 0
                    breakaway_candidate = None

                ramp_step += 1
                if ramp_step % 10 == 0:
                    _check_errors(axis)

            # Zero and rest
            _safe_torque(axis, 0.0, max_torque)
            time.sleep(1.0)

            if breakaway is None:
                print(
                    f"WARNING: no motion onset detected within max_torque={max_torque:.3f} N·m"
                )
                continue

            print(f"breakaway at {breakaway:.4f} N·m")
            results.append(RampTrial(breakaway_torque=breakaway, direction=direction))

    return results


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _deduplicate(t: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Remove duplicate timestamps, keeping first occurrence."""
    _, idx = np.unique(t, return_index=True)
    return (t[idx],) + tuple(a[idx] for a in arrays)


def _velocity_model(
    t: np.ndarray, b: float, J: float, tau_c: float, tau_step: float
) -> np.ndarray:
    """First-order velocity response to a torque step.

    tau_c and tau_step are always supplied as known constants (not fitted).
    Only b and J are free parameters when this function is used with curve_fit
    inside fit_inertia_damping.
    """
    omega_ss = (tau_step - tau_c) / b
    tau_sys = J / b
    return omega_ss * (1.0 - np.exp(-t / tau_sys))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_SECTION_WIDTH = 60


def _header(title: str) -> str:
    return f"\n{'─' * _SECTION_WIDTH}\n{title}\n{'─' * _SECTION_WIDTH}"


def _noisy_value(nominal: float, half_range: float) -> str:
    return f"NoisyValue({nominal:.4f}, {half_range:.4f})"


def _dr_range(nominal: float, spread: float, floor_pct: float = 0.0) -> tuple[float, float]:
    half = max(spread / 2.0, nominal * floor_pct)
    return max(0.0, nominal - half), nominal + half
