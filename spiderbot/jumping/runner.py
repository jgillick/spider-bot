import os
import re
import sys
import subprocess
from dataclasses import dataclass, field
from os import path
from pathlib import Path
from typing import Literal

import numpy as np

THIS_DIR = path.dirname(path.abspath(__file__))

_REWARD_RE = re.compile(r"Mean reward:\s*([-\d.eE+]+)")

# Minimum reward-curve samples before running early-stop checks.
# Avoids false flatlines during the warmup window when no episodes have
# completed yet.
_MIN_SAMPLES_FOR_STOP = 50


@dataclass
class TrainingResult:
    run_name: str
    log_dir: str
    reward_curve: list[float] = field(default_factory=list)
    exit_code: int = 0
    stop_reason: Literal["completed", "flatline", "divergence", "error"] = "completed"
    final_mean_reward: float = 0.0
    iteration_reached: int = 0
    error_tail: str = ""  # last lines of stdout when exit_code != 0


def run_training(
    iter_dir: str | Path,
    run_name: str,
    num_envs: int,
    max_iterations: int,
    device: str = "gpu",
    early_stop: bool = True,
) -> TrainingResult:
    """
    Launch a training subprocess for one run inside an experiment directory.

    iter_dir: absolute path to an experiments/iter<n>_<ts>/ directory
    run_name: "0_probe" or "1_full" — used as the experiment name inside train.py
              so logs land at <iter_dir>/logs/<run_name>/
    """
    iter_dir = Path(iter_dir)
    iter_dir_name = iter_dir.name

    log_dir = str(iter_dir / "logs" / run_name)

    module_path = f"spiderbot.jumping.experiments.{iter_dir_name}.train"
    cmd = [
        sys.executable, "-m", module_path,
        "-n", str(num_envs),
        "-i", str(max_iterations),
        "-e", run_name,
        "-d", device,
    ]

    stdout_log_path = str(iter_dir / f"{run_name}_stdout.log")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    result = TrainingResult(run_name=run_name, log_dir=log_dir)
    all_lines: list[str] = []

    try:
        with open(stdout_log_path, "w") as log_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                all_lines.append(line)

                m = _REWARD_RE.search(line)
                if m:
                    try:
                        result.reward_curve.append(float(m.group(1)))
                        result.iteration_reached = len(result.reward_curve)
                    except ValueError:
                        pass

                if early_stop and len(result.reward_curve) >= _MIN_SAMPLES_FOR_STOP:
                    reason = _check_early_stop(result.reward_curve, max_iterations)
                    if reason:
                        result.stop_reason = reason
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        break
    finally:
        proc.wait()

    result.exit_code = proc.returncode
    if result.reward_curve:
        result.final_mean_reward = result.reward_curve[-1]

    if result.stop_reason == "completed":
        if result.exit_code != 0:
            result.stop_reason = "error"
            # Capture the last 20 lines for error feedback to the LLM
            tail = "".join(all_lines[-20:]).strip()
            # Strip ANSI escape codes for readability
            tail = re.sub(r"\x1b\[[0-9;]*m", "", tail)
            result.error_tail = tail

    return result


def _check_early_stop(
    curve: list[float],
    max_iterations: int,
) -> Literal["flatline", "divergence"] | None:
    n = len(curve)
    # Only start checking after the first quarter of expected iterations
    if n < max(max_iterations // 4, _MIN_SAMPLES_FOR_STOP):
        return None

    window = max(10, n // 4)
    tail = curve[-window:]

    # Flatline: reward not moving
    if np.std(tail) < 0.05:
        return "flatline"

    # Divergence: reward peaked then collapsed — only meaningful when the
    # peak was positive (negative-only curves are caught by flatline instead)
    first_half = curve[: n // 2]
    peak = max(first_half) if first_half else 0.0
    if peak > 0.0:
        tail_mean = float(np.mean(tail))
        if tail_mean < 0.5 * peak:
            return "divergence"

    return None
