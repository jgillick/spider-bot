"""
Tests for hardware/sysid/motor.py — uses mock axis objects, no hardware.
"""

import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spider-bot"))

from motor.sysid.motor import (
    cleanup,
    decode_axis_error,
    initialize,
    poll_errors,
)
from motor.sysid.__main__ import _warn_if_anticogging_enabled


# ---------------------------------------------------------------------------
# decode_axis_error
# ---------------------------------------------------------------------------


class TestDecodeAxisError:
    def test_zero_returns_no_error(self):
        assert decode_axis_error(0) == "NO_ERROR"

    def test_known_single_flag(self):
        assert decode_axis_error(0x40) == "MOTOR_FAILED"

    def test_multiple_flags_joined(self):
        result = decode_axis_error(0x40 | 0x100)
        assert "MOTOR_FAILED" in result
        assert "ENCODER_FAILED" in result
        assert "|" in result

    def test_unknown_code_returns_hex(self):
        # Use bits 24-31 — none of the known ODrive error flags occupy this range
        result = decode_axis_error(0xFF000000)
        assert "UNKNOWN_ERROR" in result
        assert "FF000000" in result.upper()


# ---------------------------------------------------------------------------
# poll_errors
# ---------------------------------------------------------------------------


def _make_clean_axis() -> MagicMock:
    """Return a mock axis with all error registers clear."""
    axis = MagicMock()
    axis.error = 0
    axis.motor.error = 0
    axis.encoder.error = 0
    axis.controller.error = 0
    return axis


class TestPollErrors:
    def test_all_clear_returns_none(self):
        axis = _make_clean_axis()
        assert poll_errors(axis) is None

    def test_axis_error_detected(self):
        axis = _make_clean_axis()
        axis.error = 0x40  # MOTOR_FAILED
        result = poll_errors(axis)
        assert result is not None
        assert "MOTOR_FAILED" in result

    def test_motor_sub_error_detected(self):
        """Sub-register faults are caught even when axis.error is 0."""
        axis = _make_clean_axis()
        axis.motor.error = 0x100  # ENCODER_FAILED in sub-register
        result = poll_errors(axis)
        assert result is not None

    def test_encoder_sub_error_detected(self):
        axis = _make_clean_axis()
        axis.encoder.error = 0x200  # CONTROLLER_FAILED in sub-register
        result = poll_errors(axis)
        assert result is not None

    def test_controller_sub_error_detected(self):
        axis = _make_clean_axis()
        axis.controller.error = 0x40
        result = poll_errors(axis)
        assert result is not None

    def test_communication_error_on_read_failure(self):
        axis = MagicMock()
        type(axis).error = PropertyMock(side_effect=RuntimeError("USB gone"))
        axis.motor.error = 0
        axis.encoder.error = 0
        axis.controller.error = 0
        result = poll_errors(axis)
        assert result is not None
        assert "COMMUNICATION_ERROR" in result

    def test_partial_read_failure_still_reports_other_registers(self):
        """If one register can't be read, the others are still checked."""
        axis = _make_clean_axis()
        type(axis.motor).error = PropertyMock(side_effect=RuntimeError("USB gone"))
        axis.encoder.error = 0x100
        result = poll_errors(axis)
        assert result is not None  # encoder error should still be reported


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_swallows_exception_on_torque_write(self):
        axis = MagicMock()
        axis.controller.input_torque = 0.0
        type(axis).requested_state = PropertyMock(side_effect=RuntimeError("USB gone"))
        # Should not raise
        cleanup(axis)

    def test_swallows_exception_on_both_writes(self):
        axis = MagicMock()
        # Make every attribute access raise
        axis.controller.input_torque = MagicMock(side_effect=RuntimeError("gone"))
        cleanup(axis)  # must not propagate


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_raises_on_axis_error_after_init(self):
        axis = _make_clean_axis()
        # Simulate a fault discovered after state transition
        with patch("motor.sysid.motor.poll_errors", return_value="MOTOR_FAILED"):
            with pytest.raises(RuntimeError, match="Axis error after initialization"):
                initialize(axis, max_torque=1.0)

    def test_calls_cleanup_on_axis_error(self):
        axis = _make_clean_axis()
        with patch("motor.sysid.motor.poll_errors", return_value="MOTOR_FAILED"):
            with patch("motor.sysid.motor.cleanup") as mock_cleanup:
                with pytest.raises(RuntimeError):
                    initialize(axis, max_torque=1.0)
                mock_cleanup.assert_called_once_with(axis)

    def test_success_prints_confirmation(self, capsys):
        axis = _make_clean_axis()
        with patch("motor.sysid.motor.poll_errors", return_value=None):
            with patch("motor.sysid.motor.time.sleep"):
                initialize(axis, max_torque=1.0)
        out = capsys.readouterr().out
        assert "Motor initialized" in out

    def test_timeout_calls_cleanup_and_reraises(self):
        """When the state transition hangs, cleanup is called and TimeoutError propagates."""
        axis = _make_clean_axis()
        with patch(
            "motor.sysid.motor._run_with_timeout",
            side_effect=TimeoutError("timed out"),
        ):
            with patch("motor.sysid.motor.cleanup") as mock_cleanup:
                with patch("motor.sysid.motor.time.sleep"):
                    with pytest.raises(TimeoutError):
                        initialize(axis, max_torque=1.0)
                mock_cleanup.assert_called_once_with(axis)


class TestWarnIfAnticoggingEnabled:
    def test_warns_when_anticogging_enabled(self, capsys):
        axis = MagicMock()
        axis.controller.config.anticogging.anticogging_enabled = True
        _warn_if_anticogging_enabled(axis)
        captured = capsys.readouterr()
        assert "ANTICOGGING" in captured.err
        assert "anticogging_enabled" in captured.err

    def test_silent_when_anticogging_disabled(self, capsys):
        axis = MagicMock()
        axis.controller.config.anticogging.anticogging_enabled = False
        _warn_if_anticogging_enabled(axis)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_silent_when_attribute_missing(self, capsys):
        """Older ODrive firmware that doesn't expose anticogging config."""
        # Use spec=[] so accessing any attribute raises AttributeError.
        axis = MagicMock()
        axis.controller.config.anticogging = MagicMock(spec=[])
        _warn_if_anticogging_enabled(axis)
        captured = capsys.readouterr()
        assert captured.err == ""
