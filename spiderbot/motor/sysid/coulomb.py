"""
Coulomb (kinetic) friction experiment.

    run_coulomb_experiment  →  fit_coulomb  →  format_coulomb_result

Experiment: slowly ramp input_torque from zero until the output shaft first
exceeds the motion threshold. The torque at onset is the kinetic (Coulomb)
friction τ_c. Run in both directions — friction is often asymmetric.

Parameter → sim mapping:
    τ_c  → frictionloss   MuJoCo <joint frictionloss=>,
                          URDF <dynamics friction=>,
                          genesis-forge frictionloss
"""

from __future__ import annotations

import numpy as np

from .utils import (
    RampTrial,
    _MOTION_THRESHOLD,
    _dr_range,
    _header,
    _noisy_value,
    _run_ramp_experiment,
)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_coulomb_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    ramp_rate: float = 0.01,  # N·m per second — slow ramp captures kinetic onset
    motion_threshold: float = _MOTION_THRESHOLD,
    consecutive_required: int = 15,
) -> list[RampTrial]:
    """Slowly ramp input_torque and detect Coulomb (kinetic) friction onset.

    Returns a RampTrial per completed trial (CW and CCW).

    ``consecutive_required`` sets how many consecutive samples above
    ``motion_threshold`` are needed before breakaway is declared.  The default
    of 3 filters ODrive vel_estimate noise spikes that can trigger false
    breakaway at rest.
    """
    if max_torque <= 0:
        raise ValueError(f"max_torque must be > 0, got {max_torque}")
    return _run_ramp_experiment(
        axis,
        gear_ratio,
        max_torque,
        trials,
        poll_interval,
        ramp_rate,
        motion_threshold,
        trial_label="Coulomb",
        consecutive_required=consecutive_required,
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_coulomb(trials: list[RampTrial]) -> tuple[float, float, float, float]:
    """Fit Coulomb friction from ramp trials.

    Returns (tau_c_cw, tau_c_ccw, spread_cw, spread_ccw).
    Use spread / 2 as the half-range for domain randomisation.
    """
    cw = [t.breakaway_torque for t in trials if t.direction == 1]
    ccw = [t.breakaway_torque for t in trials if t.direction == -1]

    if not cw or not ccw:
        raise ValueError(
            f"Need trials in both directions; got CW={len(cw)}, CCW={len(ccw)}"
        )

    return (
        float(np.median(cw)),
        float(np.median(ccw)),
        float(np.max(cw) - np.min(cw)),  # spread = max - min
        float(np.max(ccw) - np.min(ccw)),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_coulomb_result(
    tau_c_cw: float,
    tau_c_ccw: float,
    spread_cw: float,
    spread_ccw: float,
) -> str:
    """Format Coulomb friction results for terminal output."""
    # DR floor: ±50% of nominal (Coulomb is the biggest sim-to-real killer)
    floor_pct = 0.50
    cw_lo, cw_hi = _dr_range(tau_c_cw, spread_cw, floor_pct)
    ccw_lo, ccw_hi = _dr_range(tau_c_ccw, spread_ccw, floor_pct)
    tau_c_nominal = (tau_c_cw + tau_c_ccw) / 2.0
    half_range = max((spread_cw + spread_ccw) / 4.0, tau_c_nominal * floor_pct)

    lines = [_header("Coulomb Friction (τ_c)")]
    lines.append(
        f"  CW:  {tau_c_cw:.4f} N·m  (spread: {spread_cw:.4f})  "
        f"→ DR range: ({cw_lo:.4f}, {cw_hi:.4f})"
    )
    lines.append(
        f"  CCW: {tau_c_ccw:.4f} N·m  (spread: {spread_ccw:.4f})  "
        f"→ DR range: ({ccw_lo:.4f}, {ccw_hi:.4f})"
    )
    lines.append("  Note: a single scalar nominal is used; average of CW and CCW.")
    lines.append("")
    lines.append("  MuJoCo MJCF:")
    lines.append(f'    <joint ... frictionloss="{tau_c_nominal:.4f}"/>')
    lines.append("")
    lines.append("  URDF <dynamics>:")
    lines.append(f'    <dynamics friction="{tau_c_nominal:.4f}"/>')
    lines.append("")
    lines.append("  genesis-forge ActuatorManager:")
    lines.append(f"    frictionloss={_noisy_value(tau_c_nominal, half_range)}")
    return "\n".join(lines)
