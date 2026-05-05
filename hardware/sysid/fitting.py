"""
Pure fitting functions for sysid experiments.

All functions take raw trial data and return fitted parameter values.
No hardware calls, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


@dataclass
class StepTrial:
    """Raw data from a torque-step experiment."""
    t: np.ndarray        # seconds from step onset
    vel: np.ndarray      # output-shaft rad/s (gear-ratio corrected)
    iq: np.ndarray       # Iq_measured A (sanity check only)


@dataclass
class RampTrial:
    """Result from a torque-ramp (Coulomb / static) experiment."""
    breakaway_torque: float   # motor-side N·m at which motion onset detected
    direction: int            # +1 (CW) or -1 (CCW)


def _deduplicate(t: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Remove duplicate timestamps, keeping first occurrence."""
    _, idx = np.unique(t, return_index=True)
    return (t[idx],) + tuple(a[idx] for a in arrays)


def _velocity_model(t: np.ndarray, b: float, J: float, tau_c: float, tau_step: float) -> np.ndarray:
    omega_ss = (tau_step - tau_c) / b
    tau_sys = J / b
    return omega_ss * (1.0 - np.exp(-t / tau_sys))


def fit_inertia_damping(
    trials: list[StepTrial],
    tau_step: float,
    tau_c_estimate: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Fit rotor inertia J and viscous damping b from torque-step trials.

    Returns (J_nominal, b_nominal, J_spread, b_spread) where spread = max - min.
    """
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
            continue

    if not J_vals:
        raise ValueError("curve_fit failed to converge for all inertia trials")

    J_arr = np.array(J_vals)
    b_arr = np.array(b_vals)
    return (
        float(np.median(J_arr)),
        float(np.median(b_arr)),
        float(np.ptp(J_arr)),   # max - min
        float(np.ptp(b_arr)),
    )


def fit_coulomb(trials: list[RampTrial]) -> tuple[float, float, float, float]:
    """
    Fit Coulomb friction from ramp trials.

    Returns (tau_c_cw, tau_c_ccw, spread_cw, spread_ccw).
    """
    cw = [t.breakaway_torque for t in trials if t.direction == 1]
    ccw = [t.breakaway_torque for t in trials if t.direction == -1]

    if not cw or not ccw:
        raise ValueError(f"Need trials in both directions; got CW={len(cw)}, CCW={len(ccw)}")

    return (
        float(np.median(cw)),
        float(np.median(ccw)),
        float(np.ptp(cw)),
        float(np.ptp(ccw)),
    )


def fit_static(trials: list[RampTrial]) -> tuple[float, float]:
    """
    Fit static (breakaway) friction from ramp trials.

    Returns (tau_static_cw, tau_static_ccw).
    """
    cw = [t.breakaway_torque for t in trials if t.direction == 1]
    ccw = [t.breakaway_torque for t in trials if t.direction == -1]

    if not cw or not ccw:
        raise ValueError(f"Need trials in both directions; got CW={len(cw)}, CCW={len(ccw)}")

    return float(np.median(cw)), float(np.median(ccw))


def _ramp_target(t: float, ramp_time: float, p_des: float) -> float:
    if ramp_time <= 0:
        return p_des
    return p_des * min(t / ramp_time, 1.0)


def fit_kp_kd(
    trials: list[StepTrial],
    p_des: float,
    J: float,
    b: float,
    kp_test: float,
    kd_test: float,
    ramp_time: float = 0.0,
) -> tuple[float, float]:
    """
    Fit kp and kd from MIT impedance ramp-response trials.

    The simulated response integrates J*α = kp*(p_des(t)-p) + kd*(0-v) - b*v
    where p_des(t) follows the same linear ramp used during data collection.
    J and b from the inertia fit are fixed inputs.

    Returns (kp_fitted, kd_fitted).
    """

    def simulate_response(t_eval: np.ndarray, kp: float, kd: float) -> np.ndarray:
        def ode(t: float, y: list[float]) -> list[float]:
            pos, vel = y
            p_des_t = _ramp_target(t, ramp_time, p_des)
            torque = kp * (p_des_t - pos) + kd * (0.0 - vel) - b * vel
            alpha = torque / J
            return [vel, alpha]

        sol = solve_ivp(
            ode,
            [t_eval[0], t_eval[-1]],
            [0.0, 0.0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-4,
        )
        return np.interp(t_eval, sol.t, sol.y[1])  # return velocity trajectory

    kp_vals: list[float] = []
    kd_vals: list[float] = []

    for trial in trials:
        t, vel = _deduplicate(trial.t, trial.vel)[:2]
        if len(t) < 4:
            continue

        def model(t_arr: np.ndarray, kp: float, kd: float) -> np.ndarray:
            return simulate_response(t_arr, kp, kd)

        try:
            popt, _ = curve_fit(
                model,
                t,
                vel,
                p0=[kp_test, kd_test],
                bounds=([0.0, 0.0], [1000.0, 50.0]),
                maxfev=1000,
            )
            kp_vals.append(popt[0])
            kd_vals.append(popt[1])
        except RuntimeError:
            continue

    if not kp_vals:
        raise ValueError("curve_fit failed to converge for all kp/kd trials")

    return float(np.median(kp_vals)), float(np.median(kd_vals))
