"""
Tests for hardware/sysid/output.py — format/snapshot tests, no hardware required.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hardware.sysid.output import (
    format_coulomb_result,
    format_inertia_result,
    format_kp_kd_result,
    format_static_result,
)


def _section_order(out: str, *labels: str) -> bool:
    """Return True if labels appear in the given order within out."""
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
        assert _section_order(out, "MuJoCo MJCF:", "URDF <dynamics>:", "genesis-forge ActuatorManager:")

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


class TestFormatCoulombResult:
    def test_output_order_mujoco_urdf_genesis(self):
        out = format_coulomb_result(0.142, 0.119, 0.022, 0.023)
        assert _section_order(out, "MuJoCo MJCF:", "URDF <dynamics>:", "genesis-forge ActuatorManager:")

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


class TestFormatKpKdResult:
    def test_output_order_mujoco_urdf_genesis(self):
        out = format_kp_kd_result(40.0, 1.2)
        assert _section_order(out, "MuJoCo MJCF:", "URDF:", "genesis-forge ActuatorManager:")

    def test_mujoco_snippet(self):
        out = format_kp_kd_result(40.0, 1.2)
        assert "MuJoCo MJCF:" in out
        assert 'kp="40.0000"' in out
        assert 'kv="1.2000"' in out

    def test_urdf_no_snippet(self):
        out = format_kp_kd_result(40.0, 1.2)
        assert "URDF:" in out
        # URDF section explains there is no equivalent — no XML tag to copy
        urdf_section = out[out.find("URDF:"):]
        genesis_start = urdf_section.find("genesis-forge")
        urdf_block = urdf_section[:genesis_start]
        assert "<" not in urdf_block

    def test_genesis_snippet(self):
        out = format_kp_kd_result(40.0, 1.2)
        assert "genesis-forge ActuatorManager:" in out
        assert "kp=" in out
        assert "kv=" in out
        assert "NoisyValue(" in out

    def test_dr_range_is_15_pct(self):
        out = format_kp_kd_result(40.0, 1.2)
        assert "NoisyValue(40.0000, 6.0000)" in out
        assert "NoisyValue(1.2000, 0.1800)" in out
