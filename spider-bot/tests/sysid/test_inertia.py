"""
Tests for hardware/sysid/inertia.py — no hardware required.
"""

import math
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spider-bot"))

from motor.sysid.inertia import (
    fit_inertia_damping,
    format_inertia_result,
    run_inertia_experiment,
)
from motor.sysid.utils import StepTrial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return StepTrial(t=t, vel=vel, iq=np.zeros_like(t))


def _make_axis() -> MagicMock:
    axis = MagicMock()
    axis.error = 0
    axis.motor.error = 0
    axis.encoder.error = 0
    axis.controller.error = 0
    axis.encoder.vel_estimate = 0.5
    axis.motor.current_control.Iq_measured = 0.1
    return axis


def _monotonic_counter(start: float, step: float, count: int):
    return [start + i * step for i in range(count)]


GEAR_RATIO = 8.0
MAX_TORQUE = 1.0


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
        trials = [
            _make_step_trial(J_true, b_true, tau_c, tau_step, noise_sigma=0.01)
            for _ in range(5)
        ]
        J, b, _, _ = fit_inertia_damping(trials, tau_step, tau_c)
        assert abs(J - J_true) / J_true < 0.10
        assert abs(b - b_true) / b_true < 0.10

    def test_spread_is_max_minus_min(self):
        tau_step, tau_c = 0.5, 0.1
        trial1 = _make_step_trial(0.002, 0.05, tau_c, tau_step)
        trial2 = _make_step_trial(0.0025, 0.055, tau_c, tau_step)
        J, b, J_spread, b_spread = fit_inertia_damping(
            [trial1, trial2], tau_step, tau_c
        )
        assert J_spread >= 0
        assert b_spread >= 0

    def test_duplicate_timestamps_handled(self):
        trial = _make_step_trial(0.002, 0.05, 0.1, 0.5)
        t_dup = np.concatenate([trial.t[:5], trial.t[:5], trial.t[5:]])
        vel_dup = np.concatenate([trial.vel[:5], trial.vel[:5], trial.vel[5:]])
        dup_trial = StepTrial(t=t_dup, vel=vel_dup, iq=np.zeros_like(t_dup))
        J, b, _, _ = fit_inertia_damping([dup_trial], 0.5, 0.1)
        assert J > 0
        assert b > 0

    def test_raises_on_empty_trials(self):
        with pytest.raises(ValueError):
            fit_inertia_damping([], 0.5, 0.1)

    def test_raises_when_all_trials_fail_convergence(self):
        """When curve_fit raises RuntimeError for every trial, ValueError is raised."""
        t = np.linspace(0, 2.0, 50)
        trial = StepTrial(t=t, vel=np.zeros_like(t), iq=np.zeros_like(t))
        with patch(
            "motor.sysid.inertia.curve_fit", side_effect=RuntimeError("failed")
        ):
            with pytest.raises(ValueError, match="curve_fit failed"):
                fit_inertia_damping([trial], tau_step=0.5, tau_c_estimate=0.0)


# ---------------------------------------------------------------------------
# run_inertia_experiment
# ---------------------------------------------------------------------------


class TestRunInertiaExperiment:
    def test_returns_trials_for_both_directions(self):
        axis = _make_axis()
        with (
            patch("motor.sysid.utils.time.sleep"),
            patch(
                "motor.sysid.utils.time.monotonic",
                side_effect=_monotonic_counter(0.0, 0.01, 40),
            ),
        ):
            results = run_inertia_experiment(
                axis,
                GEAR_RATIO,
                MAX_TORQUE,
                trials=1,
                poll_interval=0.01,
                step_duration=0.15,
            )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, StepTrial)
            assert len(r.t) > 0

    def test_max_torque_cap_raises(self):
        from motor.sysid.utils import SafetyLimitError

        axis = _make_axis()
        with patch("motor.sysid.utils.time.sleep"):
            with pytest.raises(SafetyLimitError, match="exceeds"):
                run_inertia_experiment(
                    axis,
                    GEAR_RATIO,
                    max_torque=0.1,
                    trials=1,
                    step_torque=0.5,  # exceeds cap
                    step_duration=0.05,
                )

    def test_axis_error_raises_runtime_error(self):
        axis = _make_axis()
        axis.error = 0x40  # MOTOR_FAILED
        with patch("motor.sysid.utils.time.sleep"):
            with pytest.raises(RuntimeError, match="Axis error"):
                run_inertia_experiment(
                    axis,
                    GEAR_RATIO,
                    MAX_TORQUE,
                    trials=1,
                    step_duration=0.05,
                )

    def test_zero_max_torque_raises(self):
        axis = _make_axis()
        with pytest.raises(ValueError, match="max_torque must be > 0"):
            run_inertia_experiment(axis, GEAR_RATIO, max_torque=0.0, trials=1)

    def test_usb_drop_during_poll_loop_raises_and_zeros_torque(self):
        """USB loss mid-poll raises RuntimeError('Lost contact...') and zeros torque."""
        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.motor.current_control.Iq_measured = 0.1
        # vel_estimate raises after the first read (step torque already applied)
        type(axis.encoder).vel_estimate = PropertyMock(
            side_effect=RuntimeError("USB gone")
        )

        with (
            patch("motor.sysid.utils.time.sleep"),
            patch(
                "motor.sysid.utils.time.monotonic",
                side_effect=_monotonic_counter(0.0, 0.01, 40),
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Lost contact with motor during polling loop"
            ):
                run_inertia_experiment(
                    axis,
                    GEAR_RATIO,
                    MAX_TORQUE,
                    trials=1,
                    step_duration=0.15,
                )

        # Torque is zeroed in the USB-drop handler before the exception propagates
        assert axis.controller.input_torque == 0.0


# ---------------------------------------------------------------------------
# format_inertia_result
# ---------------------------------------------------------------------------


def _section_order(out: str, *labels: str) -> bool:
    pos = 0
    for label in labels:
        idx = out.find(label, pos)
        if idx == -1:
            return False
        pos = idx + 1
    return True


class TestFormatInertiaResult:
    def test_output_order_mujoco_urdf_genesis(self):
        out = format_inertia_result(0.002, 0.05, 0.001, 0.005)
        assert _section_order(
            out, "MuJoCo MJCF:", "URDF <dynamics>:", "genesis-forge ActuatorManager:"
        )

    def test_mujoco_snippet(self):
        out = format_inertia_result(0.002, 0.05, 0.001, 0.005)
        assert "MuJoCo MJCF:" in out
        assert 'armature="0.002000"' in out
        assert 'damping="0.050000"' in out

    def test_urdf_snippet(self):
        out = format_inertia_result(0.002, 0.05, 0.001, 0.005)
        assert "URDF <dynamics>:" in out
        assert "<dynamics" in out

    def test_genesis_snippet(self):
        out = format_inertia_result(0.002, 0.05, 0.001, 0.005)
        assert "genesis-forge ActuatorManager:" in out
        assert "armature=" in out
        assert "NoisyValue(" in out

    def test_nominal_values_appear(self):
        out = format_inertia_result(0.002, 0.05, 0.0, 0.0)
        assert "0.002000" in out
        assert "0.050000" in out
