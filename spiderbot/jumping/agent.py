"""
Orchestrator for the autonomous jumping agent.
Runs a continuous probe → promote → eval loop, using Claude to propose
reward modifications at each iteration.

Each iteration creates a self-contained experiment directory under
experiments/iter<n>_<ts>/ containing the source files, logs, reasoning.md,
metrics.json, and evaluation.md. The base/ template is never modified.
Scrapped experiments (probe failure, no checkpoint) are moved to failed/.
"""

from __future__ import annotations

import json
import logging
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from os import path
from pathlib import Path
from typing import Literal

from .config import (
    EVAL_NUM_STEPS,
    FULL_ITERATIONS,
    FULL_NUM_ENVS,
    MAX_LLM_FAILURES,
    PROBE_ITERATIONS,
    PROBE_NUM_ENVS,
    PROMISING_HEIGHT_M, PROMISING_FORCE_N
)
from .runner import run_training, TrainingResult
from .llm_engine import propose_modifications

THIS_DIR = path.dirname(path.abspath(__file__))
_JUMPING_ROOT = Path(THIS_DIR).resolve()
_EXPERIMENTS_DIR = _JUMPING_ROOT / "experiments"
_BASE_DIR = _EXPERIMENTS_DIR / "base"
_FAILED_DIR = _JUMPING_ROOT / "failed"
_RUN_HISTORY_PATH = path.join(THIS_DIR, "run_history.json")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Files the LLM may propose changes to (written into each experiment dir)
_MANAGED_FILES = ["environment.py", "rewards.py", "terminations.py", "ppo.yaml"]

@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    probe_iterations: int = PROBE_ITERATIONS
    full_iterations: int = FULL_ITERATIONS
    probe_num_envs: int = PROBE_NUM_ENVS
    full_num_envs: int = FULL_NUM_ENVS
    device: str = "gpu"
    resume: bool = False
    commentary: str = ""


class JumpingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._run_history: list[dict] = []
        self._consecutive_llm_failures = 0
        self._iteration = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self.config.resume:
            self._load_run_history()
        self._register_sigint()

        logger.info("Starting autonomous jumping agent (model=%s)", self.config.model)
        logger.info("Experiment template: %s", _BASE_DIR)

        while True:
            self._iteration += 1
            logger.info("=== Iteration %d ===", self._iteration)
            self._run_iteration()

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def _run_iteration(self) -> None:
        prompt_mode = self._determine_prompt_mode()
        source_dir = self._get_source_dir()
        current_files = self._read_files_from(source_dir)

        # 1. Ask LLM for modifications
        proposal = propose_modifications(
            current_files=current_files,
            run_history=self._run_history,
            prompt_mode=prompt_mode,
            model=self.config.model,
            commentary=self.config.commentary,
        )
        if proposal is None:
            self._consecutive_llm_failures += 1
            logger.warning(
                "LLM returned no valid proposal (failure %d/%d)",
                self._consecutive_llm_failures, MAX_LLM_FAILURES,
            )
            if self._consecutive_llm_failures >= MAX_LLM_FAILURES:
                raise RuntimeError(
                    f"LLM failed to produce valid modifications "
                    f"{MAX_LLM_FAILURES} consecutive times — aborting"
                )
            return
        self._consecutive_llm_failures = 0

        # 2. Create experiment directory (copy base, overlay managed files from source)
        iter_ts = _ts()
        iter_dir = self._create_experiment_dir(iter_ts, source_dir)
        iter_dir_name = iter_dir.name

        # 3. Write LLM proposal into the experiment dir
        self._write_proposed_files(proposal.modified_files, iter_dir)

        # 4. Write reasoning.md immediately (before training starts)
        (iter_dir / "reasoning.md").write_text(proposal.reasoning)

        logger.info("Experiment dir: %s", iter_dir_name)

        # 5. Probe run
        logger.info("Probe run: %s (%d iters)", iter_dir_name + "/0_probe", self.config.probe_iterations)
        probe_result = run_training(
            iter_dir=iter_dir,
            run_name="0_probe",
            num_envs=self.config.probe_num_envs,
            max_iterations=self.config.probe_iterations,
            device=self.config.device,
            early_stop=True,
        )
        probe_metrics = {
            "final_mean_reward": probe_result.final_mean_reward,
            "iteration_reached": probe_result.iteration_reached,
            "stop_reason": probe_result.stop_reason,
            "error_tail": probe_result.error_tail or None,
        }

        # 6. Probe promotion check
        probe_promoted = probe_result.stop_reason == "completed" or (
            probe_result.stop_reason not in ("flatline", "divergence", "error")
            and probe_result.final_mean_reward > 0
        )

        if not probe_promoted:
            logger.info(
                "Probe scrapped: %s (final reward %.3f)",
                probe_result.stop_reason, probe_result.final_mean_reward,
            )
            if probe_result.error_tail:
                logger.warning("Subprocess error:\n%s", probe_result.error_tail)

            self._write_evaluation_doc(iter_dir=iter_dir, probe_metrics=probe_metrics)
            self._append_history(
                iter_dir=iter_dir_name,
                run_type="probe",
                stop_reason=probe_result.stop_reason,
                is_promising_flag=False,
                metrics=probe_metrics,
                reasoning=proposal.reasoning,
            )
            _move_to_failed(iter_dir)
            logger.info("Moved scrapped experiment to failed/%s", iter_dir_name)
            return

        # 7. Full run
        logger.info("Full run: %s (%d iters)", iter_dir_name + "/1_full", self.config.full_iterations)
        full_result = run_training(
            iter_dir=iter_dir,
            run_name="1_full",
            num_envs=self.config.full_num_envs,
            max_iterations=self.config.full_iterations,
            device=self.config.device,
            early_stop=True,
        )

        # Check for checkpoint before attempting eval
        checkpoint_exists = any(iter_dir.glob("logs/1_full/model_*.pt"))

        if not checkpoint_exists:
            logger.warning("No checkpoint found after full run — skipping eval")
            self._write_evaluation_doc(
                iter_dir=iter_dir,
                probe_metrics=probe_metrics,
                full_result=full_result,
            )
            self._append_history(
                iter_dir=iter_dir_name,
                run_type="full",
                stop_reason="error",
                is_promising_flag=False,
                metrics={"error": "no_checkpoint"},
                reasoning=proposal.reasoning,
            )
            _move_to_failed(iter_dir)
            logger.info("Moved scrapped experiment to failed/%s", iter_dir_name)
            return

        # 8. Eval — subprocess so Genesis process state doesn't bleed
        logger.info("Running eval for %s", iter_dir_name)
        eval_metrics = _run_eval_subprocess(
            log_dir=str(iter_dir / "logs" / "1_full"),
            iter_dir_name=iter_dir_name,
            num_steps=EVAL_NUM_STEPS,
            device=self.config.device,
        )

        promising = is_promising(eval_metrics)
        success = eval_metrics.get("success", False)

        # 9. Write metrics.json
        all_metrics = {
            "probe": probe_metrics,
            "full": {
                "stop_reason": full_result.stop_reason,
                "final_mean_reward": full_result.final_mean_reward,
                "iteration_reached": full_result.iteration_reached,
                "error_tail": full_result.error_tail or None,
            },
            "eval": eval_metrics,
        }
        (iter_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

        # 10. Write evaluation.md
        self._write_evaluation_doc(
            iter_dir=iter_dir,
            probe_metrics=probe_metrics,
            full_result=full_result,
            eval_metrics=eval_metrics,
            promising=promising,
            success=success,
        )

        self._append_history(
            iter_dir=iter_dir_name,
            run_type="full",
            stop_reason=full_result.stop_reason,
            is_promising_flag=promising,
            metrics={**eval_metrics, "stop_reason": full_result.stop_reason},
            reasoning=proposal.reasoning,
        )

        if promising:
            logger.info("Full run is PROMISING — experiment preserved at %s", iter_dir_name)

        if success:
            logger.info(
                "SUCCESS! Policy achieves clean jump. Experiment: %s  Video: %s",
                iter_dir_name,
                eval_metrics.get("video_path"),
            )
            sys.exit(0)

    # ------------------------------------------------------------------
    # Experiment directory management
    # ------------------------------------------------------------------

    def _get_source_dir(self) -> Path:
        """
        In tune mode, return the last promising experiment dir so the LLM
        sees the files it should tune.  Otherwise return the base template.
        """
        last_promising = next(
            (e for e in reversed(self._run_history)
             if e.get("run_type") == "full" and e.get("is_promising") and e.get("iter_dir")),
            None,
        )
        if last_promising:
            candidate = _EXPERIMENTS_DIR / last_promising["iter_dir"]
            if candidate.exists():
                return candidate
        return _BASE_DIR

    def _create_experiment_dir(self, iter_ts: str, source_dir: Path) -> Path:
        """
        Create experiments/iter<n>_<ts>/ by:
          1. copying base/ (always, for train.py / eval.py / __init__.py)
          2. overlaying managed files from source_dir if source_dir differs from base
        """
        iter_dir_name = f"iter{self._iteration:04d}_{iter_ts}"
        iter_dir = _EXPERIMENTS_DIR / iter_dir_name
        shutil.copytree(str(_BASE_DIR), str(iter_dir))

        if source_dir != _BASE_DIR:
            for name in _MANAGED_FILES:
                src = source_dir / name
                if src.exists():
                    shutil.copy2(str(src), str(iter_dir / name))

        return iter_dir

    def _read_files_from(self, source_dir: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in _MANAGED_FILES:
            fpath = source_dir / name
            if fpath.exists():
                result[name] = fpath.read_text()
        return result

    def _write_proposed_files(self, modified_files: dict[str, str], target_dir: Path) -> None:
        for filename, content in modified_files.items():
            if filename not in _MANAGED_FILES:
                logger.warning("Ignoring proposed file not in managed list: %r", filename)
                continue
            target = (target_dir / filename).resolve()
            if not target.is_relative_to(target_dir):
                raise ValueError(
                    f"Safety violation: proposed file {filename!r} resolves outside {target_dir}"
                )
            target.write_text(content)
            logger.debug("Wrote %s → %s", filename, target_dir.name)

    # ------------------------------------------------------------------
    # Output documents
    # ------------------------------------------------------------------

    def _write_evaluation_doc(
        self,
        iter_dir: Path,
        probe_metrics: dict,
        full_result: TrainingResult | None = None,
        eval_metrics: dict | None = None,
        promising: bool = False,
        success: bool = False,
    ) -> None:
        lines: list[str] = []
        lines.append(f"# Evaluation — {iter_dir.name}\n\n")
        lines.append(f"**Iteration:** {self._iteration}\n\n")

        lines.append("## Probe run\n\n")
        lines.append(f"- **Stop reason:** `{probe_metrics.get('stop_reason')}`\n")
        lines.append(f"- **Final mean reward:** {probe_metrics.get('final_mean_reward', 0):.4f}\n")
        lines.append(f"- **Iterations reached:** {probe_metrics.get('iteration_reached', 0)}\n")
        if probe_metrics.get("error_tail"):
            lines.append(f"\n**Error output:**\n```\n{probe_metrics['error_tail']}\n```\n")

        if full_result is not None:
            lines.append("\n## Full run\n\n")
            lines.append(f"- **Stop reason:** `{full_result.stop_reason}`\n")
            lines.append(f"- **Final mean reward:** {full_result.final_mean_reward:.4f}\n")
            lines.append(f"- **Iterations reached:** {full_result.iteration_reached}\n")
            if full_result.error_tail:
                lines.append(f"\n**Error output:**\n```\n{full_result.error_tail}\n```\n")

        if eval_metrics is not None:
            lines.append("\n## Eval results\n\n")
            if "error" in eval_metrics:
                lines.append(f"- **Error:** {eval_metrics['error']}\n")
            else:
                lines.append(f"- **Height above resting:** {eval_metrics.get('height_above_resting_m', 0):.4f} m\n")
                lines.append(f"- **Forward distance:** {eval_metrics.get('forward_distance_m', 0):.4f} m\n")
                lines.append(f"- **Max non-feet force:** {eval_metrics.get('max_non_feet_force_N', 0):.1f} N\n")
                lines.append(f"- **Airborne steps:** {eval_metrics.get('airborne_steps', 0)} ({eval_metrics.get('airborne_fraction', 0):.1%})\n")
                lines.append(f"- **Max CoM height:** {eval_metrics.get('max_height_m', 0):.4f} m\n")
                if eval_metrics.get("video_path"):
                    lines.append(f"- **Video:** `{eval_metrics['video_path']}`\n")

            lines.append(f"\n**Promising:** {'Yes' if promising else 'No'}\n")
            lines.append(f"**Success:** {'Yes' if success else 'No'}\n")

        (iter_dir / "evaluation.md").write_text("".join(lines))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _determine_prompt_mode(self) -> Literal["explore", "tune"]:
        if not self._run_history:
            return "explore"
        last = self._run_history[-1]
        if last.get("run_type") == "full" and last.get("is_promising"):
            return "tune"
        return "explore"

    def _load_run_history(self) -> None:
        if path.exists(_RUN_HISTORY_PATH):
            with open(_RUN_HISTORY_PATH) as f:
                self._run_history = json.load(f)
            self._iteration = len(self._run_history)
            logger.info("Loaded %d history entries", len(self._run_history))

    def _save_run_history(self) -> None:
        with open(_RUN_HISTORY_PATH, "w") as f:
            json.dump(self._run_history, f, indent=2)

    def _append_history(
        self,
        iter_dir: str,
        run_type: str,
        stop_reason: str,
        is_promising_flag: bool,
        metrics: dict,
        reasoning: str,
    ) -> None:
        self._run_history.append({
            "iteration": self._iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "run_type": run_type,
            "iter_dir": iter_dir,
            "stop_reason": stop_reason,
            "is_promising": is_promising_flag,
            "metrics": metrics,
            "reasoning_summary": reasoning[:500] if reasoning else "",
        })
        self._save_run_history()

    def _register_sigint(self) -> None:
        def handler(signum, frame):
            logger.info("SIGINT received — saving history and exiting")
            self._save_run_history()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def is_promising(
    metrics: dict,
    height_threshold_m: float = PROMISING_HEIGHT_M,
    force_threshold_N: float = PROMISING_FORCE_N,
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

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _move_to_failed(iter_dir: Path) -> Path:
    """Move a scrapped experiment directory into the failed/ staging area."""
    _FAILED_DIR.mkdir(exist_ok=True)
    dest = _FAILED_DIR / iter_dir.name
    shutil.move(str(iter_dir), str(dest))
    return dest


def _run_eval_subprocess(
    log_dir: str,
    iter_dir_name: str,
    num_steps: int,
    device: str,
) -> dict:
    """
    Run the eval harness from the experiment's own eval module in a fresh
    subprocess so Genesis process-level state doesn't bleed across runs.
    Results are returned as a dict parsed from the subprocess's JSON stdout.
    """
    module_path = f"spiderbot.jumping.experiments.{iter_dir_name}.eval"
    cmd = [
        sys.executable, "-m", module_path,
        log_dir,
        "--num-steps", str(num_steps),
        "--device", device,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()[-500:]
            logger.error("Eval subprocess failed (exit %d):\n%s", result.returncode, err)
            return {"error": err, "success": False}

        # The last line of stdout is the JSON result; earlier lines are Genesis logs
        json_line = result.stdout.strip().rsplit("\n", 1)[-1]
        return json.loads(json_line)
    except subprocess.TimeoutExpired:
        logger.error("Eval subprocess timed out after 600 s")
        return {"error": "timeout", "success": False}
    except Exception as exc:
        logger.error("Eval subprocess error: %s", exc)
        return {"error": str(exc), "success": False}
