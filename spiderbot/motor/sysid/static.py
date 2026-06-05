"""
Static (breakaway) friction experiment.

    run_static_experiment  →  fit_static  →  format_static_result

Experiment: ramp torque from rest using a faster rate than the Coulomb
experiment to capture the stiction peak before the motor transitions to
kinetic sliding. The measured breakaway torque should be ≥ Coulomb friction.

MuJoCo, URDF, and genesis-forge do not model static friction separately from
Coulomb; use this result to sanity-check the Coulomb measurement and to set
the DR floor for frictionloss.
"""

from __future__ import annotations

import numpy as np

from .utils import RampTrial, _MOTION_THRESHOLD, _header, _run_ramp_experiment


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_static_experiment(
    axis,
    gear_ratio: float,
    max_torque: float,
    trials: int = 5,
    poll_interval: float = 0.01,
    ramp_rate: float = 0.02,  # N·m per second — faster than Coulomb to capture stiction peak
    motion_threshold: float = _MOTION_THRESHOLD,
    consecutive_required: int = 15,
) -> list[RampTrial]:
    """Ramp from rest and detect static (stiction) breakaway torque.

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
        trial_label="Static",
        consecutive_required=consecutive_required,
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_static(trials: list[RampTrial]) -> tuple[float, float]:
    """Fit static (breakaway) friction from ramp trials.

    Returns (tau_static_cw, tau_static_ccw).
    """
    cw = [t.breakaway_torque for t in trials if t.direction == 1]
    ccw = [t.breakaway_torque for t in trials if t.direction == -1]

    if not cw or not ccw:
        raise ValueError(
            f"Need trials in both directions; got CW={len(cw)}, CCW={len(ccw)}"
        )

    return float(np.median(cw)), float(np.median(ccw))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_static_result(
    tau_static_cw: float,
    tau_static_ccw: float,
) -> str:
    """Format static (breakaway) friction results for terminal output."""
    lines = [_header("Static Friction (τ_static)")]
    lines.append(f"  CW:  {tau_static_cw:.4f} N·m")
    lines.append(f"  CCW: {tau_static_ccw:.4f} N·m")
    lines.append("")
    lines.append("  Static friction sets the minimum torque needed to break from rest.")
    lines.append(
        "  It should be ≥ Coulomb friction — use it to sanity-check that result."
    )
    lines.append(
        "  MuJoCo, URDF, and genesis-forge do not model static friction separately;"
    )
    lines.append("  no copy-paste snippet is provided.")
    return "\n".join(lines)
