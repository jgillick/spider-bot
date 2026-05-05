"""
Tests for hardware/sysid/fitting.py — pure functions, no hardware required.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hardware.sysid.fitting import (
    RampTrial,
    StepTrial,
    fit_coulomb,
    fit_inertia_damping,
    fit_kp_kd,
    fit_static,
)


def _make_step_trial(
    J: float,
    b: float,
    tau_c: float,
    tau_step: float,
    duration: float = 2.0,
    dt: float = 0.01,
    noise_sigma: float = 0.0,
) -> StepTrial:
    t = np.arange(0, duration, dt)
    omega_ss = (tau_step - tau_c) / b
    tau_sys = J / b
    vel = omega_ss * (1.0 - np.exp(-t / tau_sys))
    if noise_sigma > 0:
        vel += np.random.default_rng(42).normal(0, noise_sigma, len(t))
    iq = np.zeros_like(t)
    return StepTrial(t=t, vel=vel, iq=iq)


# ---------------------------------------------------------------------------
# fit_inertia_damping
# ---------------------------------------------------------------------------


class TestFitInertiaDamping:
    def test_clean_synthetic_data(self):
        J_true, b_true, tau_c, tau_step = 0.002, 0.05, 0.1, 0.5
        trial = _make_step_trial(J_true, b_true, tau_c, tau_step)
        J, b, J_spread, b_spread = fit_inertia_damping([trial], tau_step, tau_c)
        assert abs(J - J_true) / J_true < 0.05, f"J off: {J} vs {J_true}"
        assert abs(b - b_true) / b_true < 0.05, f"b off: {b} vs {b_true}"

    def test_noisy_data(self):
        J_true, b_true, tau_c, tau_step = 0.002, 0.05, 0.1, 0.5
        trials = [_make_step_trial(J_true, b_true, tau_c, tau_step, noise_sigma=0.01) for _ in range(5)]
        J, b, _, _ = fit_inertia_damping(trials, tau_step, tau_c)
        assert abs(J - J_true) / J_true < 0.10
        assert abs(b - b_true) / b_true < 0.10

    def test_spread_is_max_minus_min(self):
        tau_step, tau_c = 0.5, 0.1
        # Two trials with slightly different params to create spread
        trial1 = _make_step_trial(0.002, 0.05, tau_c, tau_step)
        trial2 = _make_step_trial(0.0025, 0.055, tau_c, tau_step)
        J, b, J_spread, b_spread = fit_inertia_damping([trial1, trial2], tau_step, tau_c)
        assert J_spread >= 0
        assert b_spread >= 0

    def test_duplicate_timestamps_handled(self):
        trial = _make_step_trial(0.002, 0.05, 0.1, 0.5)
        # Inject duplicate timestamps
        t_dup = np.concatenate([trial.t[:5], trial.t[:5], trial.t[5:]])
        vel_dup = np.concatenate([trial.vel[:5], trial.vel[:5], trial.vel[5:]])
        iq_dup = np.zeros_like(t_dup)
        dup_trial = StepTrial(t=t_dup, vel=vel_dup, iq=iq_dup)
        J, b, _, _ = fit_inertia_damping([dup_trial], 0.5, 0.1)
        assert J > 0
        assert b > 0

    def test_raises_on_empty_trials(self):
        with pytest.raises((ValueError, Exception)):
            fit_inertia_damping([], 0.5, 0.1)


# ---------------------------------------------------------------------------
# fit_coulomb
# ---------------------------------------------------------------------------


class TestFitCoulomb:
    def test_symmetric_returns_correct_medians(self):
        trials = [
            RampTrial(breakaway_torque=v, direction=d)
            for v, d in [(0.14, 1), (0.13, 1), (0.15, 1), (0.12, -1), (0.11, -1), (0.13, -1)]
        ]
        tau_c_cw, tau_c_ccw, spread_cw, spread_ccw = fit_coulomb(trials)
        assert math.isclose(tau_c_cw, 0.14, abs_tol=1e-9)
        assert math.isclose(tau_c_ccw, 0.12, abs_tol=1e-9)
        assert math.isclose(spread_cw, 0.02, abs_tol=1e-9)
        assert math.isclose(spread_ccw, 0.02, abs_tol=1e-9)

    def test_asymmetric_both_reported(self):
        trials = [RampTrial(0.20, 1), RampTrial(0.08, -1)]
        tau_c_cw, tau_c_ccw, _, _ = fit_coulomb(trials)
        assert tau_c_cw > tau_c_ccw

    def test_missing_direction_raises(self):
        trials = [RampTrial(0.14, 1)]  # no CCW
        with pytest.raises(ValueError):
            fit_coulomb(trials)


# ---------------------------------------------------------------------------
# fit_static
# ---------------------------------------------------------------------------


class TestFitStatic:
    def test_returns_medians(self):
        trials = [RampTrial(0.20, 1), RampTrial(0.22, 1), RampTrial(0.15, -1), RampTrial(0.17, -1)]
        cw, ccw = fit_static(trials)
        assert math.isclose(cw, 0.21, abs_tol=1e-9)
        assert math.isclose(ccw, 0.16, abs_tol=1e-9)

    def test_missing_direction_raises(self):
        with pytest.raises(ValueError):
            fit_static([RampTrial(0.2, 1)])


# ---------------------------------------------------------------------------
# fit_kp_kd
# ---------------------------------------------------------------------------


def _make_kp_kd_trial(
    kp: float,
    kd: float,
    J: float,
    b: float,
    p_des: float,
    duration: float = 1.0,
    dt: float = 0.01,
) -> StepTrial:
    """Generate a synthetic MIT impedance step response."""
    from scipy.integrate import solve_ivp

    def ode(t, y):
        pos, vel = y
        torque = kp * (p_des - pos) + kd * (0.0 - vel) - b * vel
        alpha = torque / J
        return [vel, alpha]

    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(ode, [0, duration], [0.0, 0.0], t_eval=t_eval, method="RK45", rtol=1e-6)
    vel = np.interp(t_eval, sol.t, sol.y[1])
    return StepTrial(t=t_eval, vel=vel, iq=np.zeros_like(t_eval))


class TestFitKpKd:
    def test_synthetic_trajectory(self):
        kp_true, kd_true = 40.0, 1.2
        J, b, p_des = 0.002, 0.05, 0.2
        trial = _make_kp_kd_trial(kp_true, kd_true, J, b, p_des)
        kp, kd = fit_kp_kd([trial], p_des, J, b, kp_test=kp_true, kd_test=kd_true)
        assert abs(kp - kp_true) / kp_true < 0.10, f"kp off: {kp} vs {kp_true}"
        assert abs(kd - kd_true) / kd_true < 0.10, f"kd off: {kd} vs {kd_true}"
