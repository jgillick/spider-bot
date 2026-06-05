"""
Tests for hardware/sysid/static.py — no hardware required.
"""

import math
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spider-bot"))

from motor.sysid.static import (
    fit_static,
    format_static_result,
    run_static_experiment,
)
from motor.sysid.utils import RampTrial


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


# ---------------------------------------------------------------------------
# fit_static
# ---------------------------------------------------------------------------


class TestFitStatic:
    def test_returns_medians(self):
        trials = [
            RampTrial(0.20, 1),
            RampTrial(0.22, 1),
            RampTrial(0.15, -1),
            RampTrial(0.17, -1),
        ]
        cw, ccw = fit_static(trials)
        assert math.isclose(cw, 0.21, abs_tol=1e-9)
        assert math.isclose(ccw, 0.16, abs_tol=1e-9)

    def test_missing_direction_raises(self):
        with pytest.raises(ValueError):
            fit_static([RampTrial(0.2, 1)])


# ---------------------------------------------------------------------------
# run_static_experiment
# ---------------------------------------------------------------------------


class TestRunStaticExperiment:
    def test_uses_static_label_in_output(self, capsys):
        """run_static_experiment prints 'Static trial', not 'Coulomb trial'."""
        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.encoder.vel_estimate = 1.0  # always above threshold → immediate breakaway
        axis.motor.current_control.Iq_measured = 0.1

        with patch("motor.sysid.utils.time.sleep"):
            run_static_experiment(
                axis,
                GEAR_RATIO,
                MAX_TORQUE,
                trials=1,
                poll_interval=0.01,
                ramp_rate=0.1,
            )

        captured = capsys.readouterr()
        assert "Static trial" in captured.out
        assert "Coulomb trial" not in captured.out

    def test_returns_ramp_trials(self):
        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.encoder.vel_estimate = 1.0  # always above threshold → immediate breakaway
        axis.motor.current_control.Iq_measured = 0.1

        with patch("motor.sysid.utils.time.sleep"):
            results = run_static_experiment(
                axis,
                GEAR_RATIO,
                MAX_TORQUE,
                trials=1,
                poll_interval=0.01,
                ramp_rate=0.1,
            )

        assert len(results) == 2  # CW + CCW
        for r in results:
            assert isinstance(r, RampTrial)

    def test_breakaway_detected_at_correct_torque(self):
        """
        Simulate a motor that breaks away at ~0.3 N·m.
        ramp_rate=0.1 N·m/s, poll_interval=0.01s → 0.001 N·m/step.
        Velocity stays zero for 299 steps then exceeds threshold for 400 steps.
        With consecutive_required=3, breakaway is recorded at the torque of the
        *first* high-velocity sample — not biased upward by the extra samples.
        A single isolated spike should NOT trigger breakaway.
        """
        target_torque = 0.3
        ramp_rate = 0.1
        poll_interval = 0.01
        threshold_turns_s = 0.05 * GEAR_RATIO / (2 * math.pi)
        steps_to_breakaway = int(target_torque / (ramp_rate * poll_interval))

        # One isolated spike at step 100 followed by zeros, then sustained motion.
        vel_sequence = (
            [0.0] * 100
            + [threshold_turns_s * 2]  # isolated spike — must NOT trigger breakaway
            + [0.0] * (steps_to_breakaway - 102)
            + [threshold_turns_s * 2] * 400  # sustained motion
        )
        vel_iter = iter(vel_sequence)

        axis = MagicMock()
        axis.error = 0
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        axis.motor.current_control.Iq_measured = 0.1
        type(axis.encoder).vel_estimate = PropertyMock(side_effect=vel_iter)

        with patch("motor.sysid.utils.time.sleep"):
            results = run_static_experiment(
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

    def test_zero_max_torque_raises(self):
        axis = _make_axis()
        with pytest.raises(ValueError, match="max_torque must be > 0"):
            run_static_experiment(axis, GEAR_RATIO, max_torque=0.0, trials=1)


# ---------------------------------------------------------------------------
# format_static_result
# ---------------------------------------------------------------------------


class TestFormatStaticResult:
    def test_both_directions_appear(self):
        out = format_static_result(0.20, 0.15)
        assert "CW:" in out
        assert "CCW:" in out
        assert "0.2000" in out
        assert "0.1500" in out

    def test_no_copy_paste_snippets(self):
        out = format_static_result(0.20, 0.15)
        assert "NoisyValue(" not in out
        assert "<joint" not in out
        assert "<dynamics" not in out
