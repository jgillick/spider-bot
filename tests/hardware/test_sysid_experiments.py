"""
Tests for hardware/sysid/experiments.py — uses a mock ODrive axis, no hardware.
"""

import sys
import os
import math
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hardware.sysid.experiments import (
    SafetyLimitError,
    _vel_to_rad_s,
    run_coulomb_experiment,
    run_inertia_experiment,
    run_kp_kd_experiment,
)
from hardware.sysid.fitting import RampTrial, StepTrial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_axis(error_code: int = 0) -> MagicMock:
    """Build a mock ODrive axis."""
    axis = MagicMock()
    axis.error = error_code
    axis.encoder.vel_estimate = 0.0
    axis.encoder.pos_estimate = 0.0
    axis.motor.current_control.Iq_measured = 0.1
    return axis


GEAR_RATIO = 8.0
MAX_TORQUE = 1.0


# ---------------------------------------------------------------------------
# Velocity conversion
# ---------------------------------------------------------------------------


def test_vel_to_rad_s_conversion():
    # 1 turn/s rotor × 2π / 8 = π/4 rad/s output
    result = _vel_to_rad_s(1.0, GEAR_RATIO)
    assert math.isclose(result, math.pi / 4, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# run_inertia_experiment
# ---------------------------------------------------------------------------


class TestRunInertiaExperiment:
    def test_returns_trials_for_both_directions(self):
        # Provide enough vel readings for the poll loop
        axis = MagicMock()
        axis.error = 0
        axis.encoder.vel_estimate = 0.5
        axis.motor.current_control.Iq_measured = 0.1

        # Use very short duration to avoid long sleeps
        with patch("hardware.sysid.experiments.time.sleep"), \
             patch("hardware.sysid.experiments.time.monotonic", side_effect=_monotonic_counter(0.0, 0.01, 40)):
            results = run_inertia_experiment(
                axis, GEAR_RATIO, MAX_TORQUE, trials=1, poll_interval=0.01, step_duration=0.15
            )

        # 1 trial × 2 directions = 2 results
        assert len(results) == 2
        for r in results:
            assert isinstance(r, StepTrial)
            assert len(r.t) > 0

    def test_max_torque_cap_raises(self):
        axis = _make_axis()
        with patch("hardware.sysid.experiments.time.sleep"):
            with pytest.raises(ValueError, match="max-torque"):
                run_inertia_experiment(
                    axis, GEAR_RATIO, max_torque=0.1, trials=1,
                    step_torque=0.5,  # exceeds cap
                    step_duration=0.05,
                )


# ---------------------------------------------------------------------------
# run_coulomb_experiment
# ---------------------------------------------------------------------------


class TestRunCoulombExperiment:
    def test_breakaway_detected_at_correct_torque(self):
        """
        Simulate a motor that starts moving when torque reaches ~0.3 N·m.
        Uses ramp_rate=0.1 N·m/s and poll_interval=0.01s → 0.001 N·m/step.
        300 steps reach 0.3 N·m; vel returns threshold on step 300.
        """
        target_torque = 0.3   # N·m
        ramp_rate = 0.1       # N·m/s
        poll_interval = 0.01  # s → 0.001 N·m/step
        # vel needs to be read in rotor turns/s; threshold = 0.05 rad/s output
        # → threshold in turns/s = 0.05 / (2π / GEAR_RATIO) = 0.05 * GEAR_RATIO / 2π ≈ 0.0637
        threshold_turns_s = 0.05 * GEAR_RATIO / (2 * math.pi)

        steps_to_breakaway = int(target_torque / (ramp_rate * poll_interval))

        # Each loop iteration reads vel_estimate once. Build a sequence that
        # returns 0.0 for the first steps_to_breakaway-1 reads, then onset.
        vel_sequence = [0.0] * (steps_to_breakaway - 1) + [threshold_turns_s * 2] * 400

        vel_iter = iter(vel_sequence)

        axis = MagicMock()
        axis.error = 0
        axis.motor.current_control.Iq_measured = 0.1
        type(axis.encoder).vel_estimate = PropertyMock(side_effect=vel_iter)

        with patch("hardware.sysid.experiments.time.sleep"):
            results = run_coulomb_experiment(
                axis, GEAR_RATIO, MAX_TORQUE, trials=1,
                poll_interval=poll_interval,
                ramp_rate=ramp_rate,
                motion_threshold=0.05,
            )

        cw = [r for r in results if r.direction == 1]
        assert len(cw) == 1
        assert abs(cw[0].breakaway_torque - target_torque) < 0.05

    def test_no_motion_produces_warning_not_crash(self, capsys):
        axis = MagicMock()
        axis.error = 0
        axis.encoder.vel_estimate = 0.0  # never moves
        axis.motor.current_control.Iq_measured = 0.1

        with patch("hardware.sysid.experiments.time.sleep"):
            results = run_coulomb_experiment(
                axis, GEAR_RATIO, max_torque=0.05, trials=1,
                poll_interval=0.01, ramp_rate=1.0,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        # No crash; results may be empty (no breakaway detected)

    def test_axis_error_raises(self):
        axis = MagicMock()
        axis.error = 0x40  # MOTOR_FAILED
        with patch("hardware.sysid.experiments.time.sleep"):
            with pytest.raises(RuntimeError, match="Axis error"):
                run_coulomb_experiment(
                    axis, GEAR_RATIO, MAX_TORQUE, trials=1,
                    poll_interval=0.01, ramp_rate=0.1,
                )


# ---------------------------------------------------------------------------
# run_kp_kd_experiment
# ---------------------------------------------------------------------------


class TestRunKpKdExperiment:
    def test_runs_both_directions(self):
        """Verify experiment runs CW and CCW and returns two trial sets."""
        axis = MagicMock()
        axis.error = 0
        axis.encoder.pos_estimate = 0.0
        axis.encoder.vel_estimate = 0.0
        axis.motor.current_control.Iq_measured = 0.1

        with patch("hardware.sysid.experiments.time.sleep"), \
             patch("hardware.sysid.experiments.time.monotonic",
                   side_effect=_monotonic_counter(0.0, 0.01, 40)):
            results, ramp_time = run_kp_kd_experiment(
                axis, GEAR_RATIO, MAX_TORQUE, kp_test=40, kd_test=1.2,
                trials=1, poll_interval=0.01, step_size_rad=0.2,
                duration=0.05,
            )

        assert len(results) == 2  # CW + CCW
        assert ramp_time > 0

    def test_mit_torque_formula(self):
        """τ = kp*(p_des-p) + kd*(0-v) for known pos/vel pair."""
        kp, kd = 40.0, 1.2
        step_size = 0.2   # output rad
        # pos_estimate in rotor turns: 0.1 turns → 0.1*2π/8 ≈ 0.0785 rad output
        pos_turns = 0.1
        vel_turns_s = 0.0
        pos_rad = pos_turns * 2 * math.pi / GEAR_RATIO
        vel_rad_s = vel_turns_s * 2 * math.pi / GEAR_RATIO
        p_des = step_size  # CW first trial

        expected_tau = kp * (p_des - pos_rad) + kd * (0.0 - vel_rad_s)
        # Use a high max_torque so the expected value is not clipped
        test_max_torque = 50.0

        axis = MagicMock()
        axis.error = 0
        axis.encoder.pos_estimate = pos_turns
        axis.encoder.vel_estimate = vel_turns_s
        axis.motor.current_control.Iq_measured = 0.1

        sent: list[float] = []

        with patch("hardware.sysid.experiments.time.sleep"), \
             patch("hardware.sysid.experiments.time.monotonic",
                   side_effect=_monotonic_counter(0.0, 0.01, 40)):
            with patch("hardware.sysid.experiments._safe_torque", side_effect=lambda ax, tau, mt: sent.append(tau)):
                run_kp_kd_experiment(
                    axis, GEAR_RATIO, test_max_torque, kp, kd,
                    trials=1, step_size_rad=step_size, duration=0.05,
                    ramp_time=0,  # disable ramp so first torque equals full formula
                )

        # First torque sent in CW direction should match formula
        assert len(sent) > 0
        assert math.isclose(sent[0], expected_tau, rel_tol=0.01), f"{sent[0]} vs {expected_tau}"

    def test_velocity_safety_cutoff_raises(self):
        axis = MagicMock()
        axis.error = 0
        axis.encoder.pos_estimate = 0.0
        axis.encoder.vel_estimate = 100.0  # rotor turns/s → output ~78 rad/s >> limit
        axis.motor.current_control.Iq_measured = 0.1

        with patch("hardware.sysid.experiments.time.sleep"), \
             patch("hardware.sysid.experiments.time.monotonic",
                   side_effect=_monotonic_counter(0.0, 0.01, 20)):
            with pytest.raises(SafetyLimitError, match="exceeded safety limit") as exc_info:
                run_kp_kd_experiment(
                    axis, GEAR_RATIO, MAX_TORQUE, kp_test=40, kd_test=1.2,
                    trials=1, step_size_rad=0.2, duration=0.5, vel_limit=5.0,
                )
        assert exc_info.value.ramp_time > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monotonic_counter(start: float, step: float, count: int):
    """Return a side_effect list for time.monotonic that increments by step."""
    values = [start + i * step for i in range(count)]
    return values
