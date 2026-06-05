"""
Rotor inertia (J) and viscous damping (b) experiment.

    run_inertia_experiment  →  fit_inertia_damping  →  format_inertia_result

Physics:
    J·a = τ_step - b·ω - τ_c·sign(ω)

    Step response: ω(t) = (τ_step - τ_c)/b · (1 - exp(-b/J · t))

    Reading the parameters:
      b  ← steady-state velocity:  ω_ss = (τ_step - τ_c) / b
      J  ← time constant:          τ_sys = J / b

Parameter → sim mapping:
    J  → armature     MuJoCo <joint armature=>,  URDF <dynamics armature=>,  genesis-forge armature
    b  → damping      MuJoCo <joint damping=>,   URDF <dynamics damping=>,   genesis-forge damping
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import curve_fit

from .utils import (
    StepTrial,
    _check_errors,
    _deduplicate,
    _dr_range,
    _header,
    _noisy_value,
    _poll_loop,
    _safe_torque,
    _velocity_model,
)


# ---------------------------------------------------------------------------
# Experiment
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
    """Apply a torque step and record the velocity response.

    Returns StepTrial list for both CW and CCW directions
    (``trials`` per direction = 2 x trials total).

    The same trial data feeds both fit_inertia_damping (J and b from curve
    fit) and the steady-state velocity check.
    """
    if max_torque <= 0:
        raise ValueError(f"max_torque must be > 0, got {max_torque}")
    if step_torque is None:
        step_torque = min(0.5, max_torque * 0.5)

    results: list[StepTrial] = []

    for direction in (1, -1):
        tau = direction * step_torque
        for trial_idx in range(trials):
            print(
                f"  Inertia trial {trial_idx + 1}/{trials} "
                f"{'CW' if direction == 1 else 'CCW'} "
                f"(τ={tau:.3f} N·m)...",
                end=" ",
                flush=True,
            )
            _check_errors(axis)

            # Apply step torque, poll velocity, then zero and coast to rest
            _safe_torque(axis, tau, max_torque)
            t_arr, vel_arr, iq_arr = _poll_loop(
                axis, gear_ratio, step_duration, poll_interval
            )
            _safe_torque(axis, 0.0, max_torque)
            time.sleep(1.0)

            print("done")
            results.append(StepTrial(t=t_arr, vel=np.abs(vel_arr), iq=iq_arr))

    return results


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_inertia_damping(
    trials: list[StepTrial],
    tau_step: float,
    tau_c_estimate: float = 0.0,
) -> tuple[float, float, float, float]:
    """Fit rotor inertia J and viscous damping b from torque-step trials.

    Returns (J_nominal, b_nominal, J_spread, b_spread) where spread = max - min.
    Use spread / 2 as the half-range for domain randomisation.
    """
    if not trials:
        raise ValueError("No trials provided")
    if tau_c_estimate >= tau_step:
        raise ValueError(
            f"tau_c_estimate ({tau_c_estimate:.4f}) must be less than tau_step ({tau_step:.4f}). "
            "When Coulomb friction equals or exceeds the step torque the motor will not accelerate."
        )
    if tau_c_estimate == 0.0:
        import sys

        print(
            "WARNING: tau_c_estimate=0.0 (Coulomb friction not measured). "
            "Run experiment 3 first to improve J and b accuracy.",
            file=sys.stderr,
        )

    J_vals: list[float] = []
    b_vals: list[float] = []

    for trial in trials:
        t, vel = _deduplicate(trial.t, trial.vel)[:2]
        if len(t) < 4:
            continue

        # Fix tau_c and tau_step as constants; free params [b, J]
        def model(t_arr: np.ndarray, b: float, J: float) -> np.ndarray:
            return _velocity_model(t_arr, b, J, tau_c_estimate, tau_step)

        try:
            popt, _ = curve_fit(
                model,
                t,
                vel,
                p0=[0.01, 0.001],
                bounds=([1e-6, 1e-6], [10.0, 1.0]),
                maxfev=5000,
            )
            b_vals.append(popt[0])
            J_vals.append(popt[1])
        except RuntimeError:
            # curve_fit raises RuntimeError when it cannot converge.
            # Other RuntimeErrors from numpy or the model closure would also be
            # swallowed here; add logging if silent failures become hard to debug.
            continue

    if not J_vals:
        raise ValueError("curve_fit failed to converge for all inertia trials")

    J_arr = np.array(J_vals)
    b_arr = np.array(b_vals)
    return (
        float(np.median(J_arr)),
        float(np.median(b_arr)),
        float(np.max(J_arr) - np.min(J_arr)),  # spread = max - min
        float(np.max(b_arr) - np.min(b_arr)),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_inertia_result(
    J: float,
    b: float,
    J_spread: float,
    b_spread: float,
) -> str:
    """Format rotor inertia and viscous damping results for terminal output."""
    J_lo, J_hi = _dr_range(J, J_spread)
    b_lo, b_hi = _dr_range(b, b_spread)

    lines = [_header("Rotor Inertia (J) + Viscous Damping (b)")]
    lines.append(
        f"  J:  {J:.6f} kg·m²     (spread: {J_spread:.6f})  "
        f"→ DR range: ({J_lo:.6f}, {J_hi:.6f})"
    )
    lines.append(
        f"  b:  {b:.6f} N·m·s/rad  (spread: {b_spread:.6f})  "
        f"→ DR range: ({b_lo:.6f}, {b_hi:.6f})"
    )
    lines.append("")
    lines.append("  MuJoCo MJCF:")
    lines.append(f'    <joint ... armature="{J:.6f}" damping="{b:.6f}"/>')
    lines.append("")
    lines.append("  URDF <dynamics>:")
    lines.append(f'    <dynamics armature="{J:.6f}" damping="{b:.6f}"/>')
    lines.append("")
    lines.append("  genesis-forge ActuatorManager:")
    lines.append(f"    armature={_noisy_value(J, J_spread / 2)}")
    lines.append(f"    damping={_noisy_value(b, b_spread / 2)}")
    return "\n".join(lines)
