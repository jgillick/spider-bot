"""
Five sysid experiment runners.

Each runner sends torque commands via the ODrive USB interface, polls encoder
feedback, and returns raw trial data. No fitting logic lives here.

Velocity conversion — encoder is rotor-side (confirmed by MOTOR_UNITS_PER_REV=8
in spider-bot/motor/protocol.py):

    vel_output_rad_s = axis.encoder.vel_estimate * 2 * pi / gear_ratio
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


class SafetyLimitError(RuntimeError):
    """Raised when a soft safety limit is exceeded during an experiment.

    These are recoverable — the motor is already zeroed before this is raised.
    ramp_time carries the ramp duration that was in use so the caller can
    double it and retry.
    """

    def __init__(self, message: str, ramp_time: float = 0.0) -> None:
        super().__init__(message)
        self.ramp_time = ramp_time

from .fitting import RampTrial, StepTrial
from .motor import cleanup, poll_errors

TWO_PI = 2.0 * math.pi

# Default motion-onset threshold for Coulomb / static ramp experiments (rad/s)
_MOTION_THRESHOLD = 0.05


def _vel_to_rad_s(vel_estimate: float, gear_ratio: float) -> float:
    return vel_estimate * TWO_PI / gear_ratio


def _pos_to_rad(pos_estimate: float, gear_ratio: float) -> float:
    return pos_estimate * TWO_PI / gear_ratio


def _safe_torque(axis, torque: float, max_torque: float) -> None:
    """Send a torque command, raising if it would exceed the cap."""
    if abs(torque) > max_torque:
        raise ValueError(
            f"Torque command {torque:.4f} N·m exceeds --max-torque {max_torque:.4f} N·m. "
            "Raise --max-torque or reduce the experiment step size."
        )
    axis.controller.input_torque = torque


def _check_errors(axis) -> None:
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
    """Poll encoder at poll_interval for duration seconds. Returns (t, vel, iq)."""
    t_list: list[float] = []
    vel_list: list[float] = []
    iq_list: list[float] = []

    t_start = time.monotonic()
    while True:
        t_now = time.monotonic() - t_start
        if t_now >= duration:
            break
        vel_list.append(_vel_to_rad_s(axis.encoder.vel_estimate, gear_ratio))
        t_list.append(t_now)
        try:
            iq_list.append(float(axis.motor.current_control.Iq_measured))
        except Exception:
            iq_list.append(float("nan"))
        time.sleep(poll_interval)

    return np.array(t_list), np.array(vel_list), np.array(iq_list)


def _detect_steady_state(
    vel: np.ndarray,
    window: int = 10,
    threshold: float = 0.005,
) -> int:
    """Return index at which velocity is considered steady-state."""
    if len(vel) < window:
        return len(vel) - 1
    for i in range(window, len(vel)):
        if np.std(vel[i - window:i]) < threshold:
            return i
    return len(vel) - 1


# ---------------------------------------------------------------------------
# IU2-A: Rotor inertia + viscous damping (shared torque step)
# ---------------------------------------------------------------------------


def run_inertia_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    step_duration: float = 3.0,
    step_torque: float | None = None,
) -> list[StepTrial]:
    """
    Apply torque step, record velocity response. Returns StepTrial list for
    both CW and CCW directions (trials per direction).

    The same trial data is used for both J (IU3: fit_inertia_damping) and
    b (steady-state read in fit_inertia_damping).
    """
    if step_torque is None:
        step_torque = min(0.5, max_torque * 0.5)

    results: list[StepTrial] = []

    for direction in (1, -1):
        tau = direction * step_torque
        for trial_idx in range(trials):
            print(f"  Inertia trial {trial_idx + 1}/{trials} {'CW' if direction == 1 else 'CCW'} "
                  f"(τ={tau:.3f} N·m)...", end=" ", flush=True)
            _check_errors(axis)

            # Step torque
            _safe_torque(axis, tau, max_torque)
            t_arr, vel_arr, iq_arr = _poll_loop(axis, gear_ratio, step_duration, poll_interval)

            # Zero and allow motor to coast to rest
            _safe_torque(axis, 0.0, max_torque)
            time.sleep(1.0)

            print("done")
            results.append(StepTrial(t=t_arr, vel=np.abs(vel_arr), iq=iq_arr))

    return results


# ---------------------------------------------------------------------------
# IU2-B: Coulomb friction — slow torque ramp until motion onset
# ---------------------------------------------------------------------------


def run_coulomb_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    ramp_rate: float = 0.01,       # N·m per second
    motion_threshold: float = _MOTION_THRESHOLD,
) -> list[RampTrial]:
    """
    Slowly ramp input_torque and detect the torque at which velocity first
    exceeds motion_threshold. Reports per direction.
    """
    results: list[RampTrial] = []

    for direction in (1, -1):
        for trial_idx in range(trials):
            print(f"  Coulomb trial {trial_idx + 1}/{trials} "
                  f"{'CW' if direction == 1 else 'CCW'}...", end=" ", flush=True)
            _check_errors(axis)

            tau = 0.0
            breakaway: float | None = None

            while abs(tau) < max_torque:
                tau += direction * ramp_rate * poll_interval
                _safe_torque(axis, tau, max_torque)
                time.sleep(poll_interval)

                vel = _vel_to_rad_s(axis.encoder.vel_estimate, gear_ratio)
                if abs(vel) >= motion_threshold:
                    breakaway = abs(tau)
                    break

            # Zero and rest
            _safe_torque(axis, 0.0, max_torque)
            time.sleep(1.0)

            if breakaway is None:
                print(f"WARNING: no motion onset detected within max_torque={max_torque:.3f} N·m")
                continue

            print(f"breakaway at {breakaway:.4f} N·m")
            results.append(RampTrial(breakaway_torque=breakaway, direction=direction))

    return results


# ---------------------------------------------------------------------------
# IU2-C: Static friction — breakaway detection from rest
# ---------------------------------------------------------------------------


def run_static_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    ramp_rate: float = 0.02,       # N·m per second (faster ramp than Coulomb)
    motion_threshold: float = _MOTION_THRESHOLD,
) -> list[RampTrial]:
    """
    Ramp from rest and detect breakaway (stiction) torque.

    Uses a faster ramp than Coulomb to capture the static friction peak before
    the motor transitions to kinetic sliding.
    """
    # Same implementation as Coulomb but with a faster ramp rate.
    # The breakaway torque from rest (stiction) will be ≥ Coulomb friction.
    return run_coulomb_experiment(
        axis,
        gear_ratio,
        max_torque,
        trials=trials,
        poll_interval=poll_interval,
        ramp_rate=ramp_rate,
        motion_threshold=motion_threshold,
    )


# ---------------------------------------------------------------------------
# IU2-D: kp/kd step response via soft MIT impedance loop
# ---------------------------------------------------------------------------


def _ramp_target(t: float, ramp_time: float, p_des: float) -> float:
    """Linearly ramp p_des from 0 to p_des over ramp_time, then hold."""
    if ramp_time <= 0:
        return p_des
    return p_des * min(t / ramp_time, 1.0)


def run_kp_kd_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    kp_test: float,
    kd_test: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    step_size_rad: float = 0.2,    # output-shaft radians
    duration: float = 1.5,
    vel_limit: float = 5.0,        # rad/s — safety cutoff
    ramp_time: float | None = None, # seconds to ramp p_des to target; None = auto
) -> tuple[list[StepTrial], float]:
    """
    Implement a soft MIT impedance loop in Python:

        τ = kp * (p_des(t) - pos) + kd * (0 - vel)

    p_des(t) is ramped linearly from 0 to step_size_rad over ramp_time, then
    held. This keeps position error — and thus torque — bounded, preventing
    velocity safety limit violations. ramp_time defaults to
    step_size_rad / (vel_limit * 0.4), which targets ~40% of vel_limit during
    the ramp.

    Runs in TORQUE_CONTROL mode. The fitted kp/kd are directly usable in
    genesis-forge ActuatorManager without unit conversion.

    Returns (trials, ramp_time_used) so the caller can pass ramp_time to
    fit_kp_kd for a consistent ODE.
    """
    if ramp_time is None:
        ramp_time = step_size_rad / (vel_limit * 0.4)

    results: list[StepTrial] = []

    for direction in (1, -1):
        p_des_rad = direction * step_size_rad
        for trial_idx in range(trials):
            print(f"  kp/kd trial {trial_idx + 1}/{trials} "
                  f"{'CW' if direction == 1 else 'CCW'} "
                  f"(p_des={p_des_rad:.3f} rad, ramp={ramp_time:.2f}s)...", end=" ", flush=True)
            _check_errors(axis)

            t_list: list[float] = []
            vel_list: list[float] = []
            iq_list: list[float] = []

            t_start = time.monotonic()
            while True:
                t_now = time.monotonic() - t_start
                if t_now >= duration:
                    break

                pos = _pos_to_rad(axis.encoder.pos_estimate, gear_ratio)
                vel = _vel_to_rad_s(axis.encoder.vel_estimate, gear_ratio)

                # Safety: abort if velocity still exceeds limit despite ramp
                if abs(vel) > vel_limit:
                    _safe_torque(axis, 0.0, max_torque)
                    raise SafetyLimitError(
                        f"Velocity {vel:.2f} rad/s exceeded safety limit {vel_limit:.2f} rad/s "
                        "during kp/kd experiment.",
                        ramp_time=ramp_time,
                    )

                p_des_t = _ramp_target(t_now, ramp_time, p_des_rad)
                tau = kp_test * (p_des_t - pos) + kd_test * (0.0 - vel)
                _safe_torque(axis, max(-max_torque, min(max_torque, tau)), max_torque)

                t_list.append(t_now)
                vel_list.append(vel)
                try:
                    iq_list.append(float(axis.motor.current_control.Iq_measured))
                except Exception:
                    iq_list.append(float("nan"))

                time.sleep(poll_interval)

            # Return to zero
            _safe_torque(axis, 0.0, max_torque)
            time.sleep(1.0)

            print("done")
            results.append(StepTrial(
                t=np.array(t_list),
                vel=np.array(vel_list),
                iq=np.array(iq_list),
            ))

    return results, ramp_time
