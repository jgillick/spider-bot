import glob
import json
import os
import shutil
from datetime import datetime
from os import path
from typing import Literal

THIS_DIR = path.dirname(path.abspath(__file__))
EXPERIMENTS_DIR = path.join(THIS_DIR, "experiments")
PRUNED_DIR = path.join(EXPERIMENTS_DIR, "pruned")


def _run_id(iteration: int, run_type: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"iter{iteration:04d}_{run_type}_{ts}"


def create_snapshot(
    iteration: int,
    run_type: Literal["probe", "full"],
    log_dir: str,
    metrics: dict,
    reasoning: str,
) -> str:
    """
    Create a snapshot directory for a probe or full training run.
    Returns the absolute path to the snapshot directory.
    """
    run_id = _run_id(iteration, run_type)
    snapshot_path = path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(snapshot_path, exist_ok=True)

    # Copy the working jumping folder state
    for filename in ("environment.py", "ppo.yaml", "train.py"):
        src = path.join(THIS_DIR, filename)
        if path.exists(src):
            shutil.copy(src, path.join(snapshot_path, filename))

    # Copy latest checkpoint for full runs
    if run_type == "full":
        checkpoint = _latest_checkpoint(log_dir)
        if checkpoint:
            shutil.copy(checkpoint, path.join(snapshot_path, path.basename(checkpoint)))

    # Write metadata
    meta = {"iteration": iteration, "run_type": run_type, "run_id": run_id, **metrics}
    with open(path.join(snapshot_path, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(path.join(snapshot_path, "reasoning.md"), "w") as f:
        f.write(reasoning)

    return snapshot_path


def prune_snapshot(snapshot_path: str) -> None:
    """
    Moves metrics.json and reasoning.md to experiments/pruned/<run_id>/,
    then deletes the snapshot directory.
    """
    run_id = path.basename(snapshot_path)
    pruned_path = path.join(PRUNED_DIR, run_id)
    os.makedirs(pruned_path, exist_ok=True)

    for filename in ("metrics.json", "reasoning.md"):
        src = path.join(snapshot_path, filename)
        if path.exists(src):
            shutil.move(src, path.join(pruned_path, filename))

    shutil.rmtree(snapshot_path, ignore_errors=True)


def is_promising(
    metrics: dict,
    height_threshold_m: float = 0.02,
    force_threshold_N: float = 15.0,
) -> bool:
    """
    Classify a full run as promising (R9):
      - CoM height above resting exceeds threshold
      - Forward distance is positive
      - Peak non-feet contact force at landing is below threshold
    """
    return (
        metrics.get("height_above_resting_m", 0.0) > height_threshold_m
        and metrics.get("forward_distance_m", 0.0) > 0.0
        and metrics.get("max_non_feet_force_N", float("inf")) < force_threshold_N
    )


def _latest_checkpoint(log_dir: str) -> str | None:
    """Return the path to the highest-iteration model_*.pt checkpoint."""
    pattern = path.join(log_dir, "model_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    def _iter(p: str) -> int:
        try:
            return int(path.splitext(path.basename(p))[0].split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=_iter)
