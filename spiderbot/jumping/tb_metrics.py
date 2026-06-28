"""
TensorBoard metric extraction for the jumping agent.

Reads event files after a training run and returns compact summary statistics
over the final quarter of training. The LLM sees these in run_history to
diagnose what the robot learned — or failed to learn.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# TensorBoard tag → compact key written into metrics.json / run_history
_TAGS = {
    "Metrics / all_feet_off_frac":  "all_feet_off_frac",
    "Metrics / feet_airborne_frac": "feet_airborne_frac",
    "Metrics / max_jump_distance":  "max_jump_distance_m",
    "Metrics / mean_flight_steps":  "mean_flight_steps",
    "Metrics / mean_forward_vel":   "mean_forward_vel",
    "Metrics / mean_upward_vel":    "mean_upward_vel",
    "Metrics / mean_landing_force": "mean_landing_force_N",
    "Metrics / clean_landing_rate": "clean_landing_rate",
    "Train/mean_reward":            "mean_reward",
}

_TAIL_FRACTION = 0.25


def extract_training_trends(log_dir: str | Path) -> dict:
    """
    Parse TensorBoard event files in log_dir and return summary statistics
    for key episode metrics over the final 25% of training iterations.

    Returns an empty dict if the event file is missing or unreadable.
    Each metric entry has:
      - tail_mean: mean over the last 25% of logged steps
      - tail_max:  max over the last 25% of logged steps
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard package not available — skipping training trend extraction")
        return {}

    log_dir = str(log_dir)
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
    except Exception as exc:
        logger.warning("Failed to load TensorBoard events from %s: %s", log_dir, exc)
        return {}

    available = set(ea.Tags().get("scalars", []))
    result: dict[str, dict] = {}

    for tag, key in _TAGS.items():
        if tag not in available:
            continue
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue

        n_tail = max(1, int(len(events) * _TAIL_FRACTION))
        tail = [e.value for e in events[-n_tail:]]

        result[key] = {
            "tail_mean": round(sum(tail) / len(tail), 4),
            "tail_max":  round(max(tail), 4),
        }

    return result
