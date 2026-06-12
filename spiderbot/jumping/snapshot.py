from os import path

THIS_DIR = path.dirname(path.abspath(__file__))
EXPERIMENTS_DIR = path.join(THIS_DIR, "experiments")


def is_promising(
    metrics: dict,
    height_threshold_m: float = 0.02,
    force_threshold_N: float = 15.0,
) -> bool:
    """
    Classify a full run as promising:
      - CoM height above resting exceeds threshold
      - Forward distance is positive
      - Peak non-feet contact force at landing is below threshold
    """
    return (
        metrics.get("height_above_resting_m", 0.0) > height_threshold_m
        and metrics.get("forward_distance_m", 0.0) > 0.0
        and metrics.get("max_non_feet_force_N", float("inf")) < force_threshold_N
    )
