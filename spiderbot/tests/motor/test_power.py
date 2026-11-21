"""Tests for spiderbot/motor/power.py."""

import pytest
import torch

from spiderbot.motor.power import copper_loss, electrical_power


def test_copper_loss_is_sign_independent():
    assert copper_loss(10.0, 0.436) == pytest.approx(65.4)
    assert copper_loss(-10.0, 0.436) == pytest.approx(65.4)


def test_electrical_power_floats():
    copper, mech = electrical_power(4.3, 2.0, torque_constant=0.43, phase_resistance=0.436)
    assert copper == pytest.approx(65.4)  # 10 A
    assert mech == pytest.approx(8.6)
    _, braking = electrical_power(4.3, -2.0, torque_constant=0.43, phase_resistance=0.436)
    assert braking == 0.0


def test_electrical_power_tensors_drop_negative_work():
    torque = torch.tensor([[4.3, -4.3], [0.0, 2.15]])
    velocity = torch.tensor([[2.0, 2.0], [5.0, -1.0]])
    copper, mech = electrical_power(torque, velocity, torque_constant=0.43, phase_resistance=0.436)
    assert torch.allclose(copper, torch.tensor([[65.4, 65.4], [0.0, 16.35]]))
    assert torch.allclose(mech, torch.tensor([[8.6, 0.0], [0.0, 0.0]]))
