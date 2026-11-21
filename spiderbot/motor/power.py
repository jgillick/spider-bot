"""
Electrical power model for the GIM8108-8 leg motors.

Used by the simulator metrics and the leg power measurement script so both report
power the same way. Torques and velocities are output-shaft side.
"""

from __future__ import annotations

# From hardware/GIM8108-8/odrive-config. Not yet checked against bench measurements
# (see spiderbot/locomotion/measure_leg_power.py).
TORQUE_CONSTANT = 0.43  # N·m/A, output side
PHASE_RESISTANCE = 0.436  # ohm
SUPPLY_VOLTAGE = 44.4  # V, 12S nominal


def copper_loss(iq, phase_resistance: float = PHASE_RESISTANCE):
    """Winding heat in watts for a peak phase current *iq* (amplitude-invariant dq)."""
    return 1.5 * phase_resistance * iq * iq


def electrical_power(
    torque,
    velocity,
    torque_constant: float = TORQUE_CONSTANT,
    phase_resistance: float = PHASE_RESISTANCE,
):
    """
    Estimate the bus power each motor draws, in watts, as ``(copper_loss, mechanical)``.

    Mechanical power counts only positive work (τ·ω > 0): energy put back while
    braking is assumed lost, which overestimates the draw a little. Works on floats
    and on torch tensors.
    """
    copper = copper_loss(torque / torque_constant, phase_resistance)
    mech = torque * velocity
    mech = mech.clamp(min=0.0) if hasattr(mech, "clamp") else max(mech, 0.0)
    return copper, mech
