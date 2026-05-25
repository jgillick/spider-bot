"""
Tests for hardware/sysid/coulomb.py — no hardware required.
"""

import math
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spider-bot"))

from motor.sysid.coulomb import (
    fit_coulomb,
    format_coulomb_result,
    run_coulomb_experiment,
)
from motor.sysid.utils import RampTrial, SafetyLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GEAR_RATIO = 8.0
MAX_TORQUE = 1.0


def _make_axis() -> MagicMock:
    axis = MagicMock()
    axis.error = 0
    axis.motor.error = 0
    axis.encoder.error = 0
    axis.controller.error = 0
    axis.encoder.vel_estimate = 0.0
    axis.motor.current_control.Iq_measured = 0.1
    return axis


def _section_order(out: str, *labels: str) -> bool:
    pos = 0
    for label in labels:
        idx = out.find(label, pos)
        if idx == -1:
            return False
        pos = idx + 1
    return True


# ---------------------------------------------------------------------------
# fit_coulomb
# ---------------------------------------------------------------------------


class TestFitCoulomb:
    def test_symmetric_returns_correct_medians(self):
        trials = [
            RampTrial(breakaway_torque=v, direction=d)
            for v, d in [
                (0.14, 1),
                (0.13, 1),
                (0.15, 1),
                (0.12, -1),
                (0.11, -1),
                (0.13, -1),
            ]
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
        with pytest.raises(ValueError):
            fit_coulomb([RampTrial(0.14, 1)])  # no CCW


# ---------------------------------------------------------------------------
# run_coulomb_experiment
# ---------------------------------------------------------------------------


class TestRunCoulombExperiment:
    def test_breakaway_detected_at_correct_torque(self):
        """
        Simulate a motor that starts moving when torque reaches ~0.3 N·m.
        ramp_rate=0.1 N·m/s, poll_interval=0.01s → 0.001 N·m/step.
        300 steps reach 0.3 N·m; vel returns above threshold on step 300.
        """
        target_torque = 0.3
        ramp_rate = 0.1
        poll_interval = 0.01
        threshold_turns_s = 0.05 * GEAR_RATIO / (2 * math.pi)
        steps_to_breakaway = int(target_torque / (ramp_rate * poll_interval))

        vel_sequence = [0.0] * (steps_to_breakaway - 1) + [threshold_turns_s * 2] * 400
        vel_iter = iter(vel_sequence)

        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.motor.current_control.Iq_measured = 0.1
        type(axis.encoder).vel_estimate = PropertyMock(side_effect=vel_iter)

        with patch("motor.sysid.utils.time.sleep"):
            results = run_coulomb_experiment(
                axis,
                GEAR_RATIO,
                MAX_TORQUE,
                trials=1,
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
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.encoder.vel_estimate = 0.0  # never moves
        axis.motor.current_control.Iq_measured = 0.1

        with patch("motor.sysid.utils.time.sleep"):
            run_coulomb_experiment(
                axis,
                GEAR_RATIO,
                max_torque=0.05,
                trials=1,
                poll_interval=0.01,
                ramp_rate=1.0,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_axis_error_raises(self):
        axis = MagicMock()
        axis.error = 0x40  # MOTOR_FAILED
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        with patch("motor.sysid.utils.time.sleep"):
            with pytest.raises(RuntimeError, match="Axis error"):
                run_coulomb_experiment(
                    axis,
                    GEAR_RATIO,
                    MAX_TORQUE,
                    trials=1,
                    poll_interval=0.01,
                    ramp_rate=0.1,
                )

    def test_zero_max_torque_raises(self):
        axis = _make_axis()
        with pytest.raises(ValueError, match="max_torque must be > 0"):
            run_coulomb_experiment(axis, GEAR_RATIO, max_torque=0.0, trials=1)

    def test_safety_limit_error_raised_when_torque_exceeds_cap(self):
        """SafetyLimitError is raised when torque exceeds cap."""
        axis = _make_axis()
        with patch("motor.sysid.utils.time.sleep"):
            with pytest.raises(SafetyLimitError):
                run_coulomb_experiment(
                    axis,
                    GEAR_RATIO,
                    max_torque=0.001,
                    trials=1,
                    poll_interval=0.01,
                    ramp_rate=10.0,  # huge ramp — exceeds cap
                )

    def test_usb_drop_during_ramp_raises_and_zeros_torque(self):
        """USB loss mid-ramp raises RuntimeError and zeros torque before re-raise."""
        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.motor.current_control.Iq_measured = 0.1
        # vel_estimate raises on the first poll (simulating USB drop during ramp)
        type(axis.encoder).vel_estimate = PropertyMock(
            side_effect=RuntimeError("USB gone")
        )

        with patch("motor.sysid.utils.time.sleep"):
            with pytest.raises(
                RuntimeError, match="Lost contact with motor during ramp"
            ):
                run_coulomb_experiment(
                    axis,
                    GEAR_RATIO,
                    MAX_TORQUE,
                    trials=1,
                    poll_interval=0.01,
                    ramp_rate=0.1,
                )

        # Torque is zeroed in the USB-drop handler before the exception propagates
        assert axis.controller.input_torque == 0.0


# ---------------------------------------------------------------------------
# format_coulomb_result
# ---------------------------------------------------------------------------


class TestFormatCoulombResult:
    def test_output_order_mujoco_urdf_genesis(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert _section_order(
            out, "MuJoCo MJCF:", "URDF <dynamics>:", "genesis-forge ActuatorManager:"
        )

    def test_mujoco_snippet(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert "MuJoCo MJCF:" in out
        assert "frictionloss=" in out

    def test_urdf_snippet(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert "URDF <dynamics>:" in out
        assert 'friction="' in out

    def test_genesis_snippet(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert "genesis-forge ActuatorManager:" in out
        assert "frictionloss=" in out
        assert "NoisyValue(" in out

    def test_dr_floor_applies_when_spread_is_small(self):
        out = format_coulomb_result(0.14, 0.14, 0.001, 0.001)
        assert "NoisyValue(0.1400, 0.0700)" in out

    def test_both_directions_appear(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert "CW:" in out
        assert "CCW:" in out
