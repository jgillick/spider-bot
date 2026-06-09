"""
Orchestrator for the autonomous jumping agent.
Runs a continuous probe → promote → eval loop, using Claude to propose
reward modifications at each iteration.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from os import path
from pathlib import Path
from typing import Literal

from . import eval as eval_harness
from .runner import run_training
from .snapshot import create_snapshot, prune_snapshot, is_promising, EXPERIMENTS_DIR
from .llm_engine import propose_modifications

THIS_DIR = path.dirname(path.abspath(__file__))
_JUMPING_ROOT = Path(THIS_DIR).resolve()
_RUN_HISTORY_PATH = path.join(THIS_DIR, "run_history.json")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Files the agent may modify and that are backed up/restored on SIGINT
_MANAGED_FILES = ["environment.py", "ppo.yaml", "train.py"]

# Maximum consecutive LLM failures before aborting
_MAX_LLM_FAILURES = 3


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    probe_iterations: int = 600
    full_iterations: int = 1500
    probe_num_envs: int = 512
    full_num_envs: int = 2096
    device: str = "gpu"


class JumpingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._run_history: list[dict] = []
        self._current_proc: subprocess.Popen | None = None
        self._consecutive_llm_failures = 0
        self._iteration = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._load_run_history()
        self._backup_managed_files()
        self._register_sigint()

        logger.info("Starting autonomous jumping agent (model=%s)", self.config.model)

        while True:
            self._iteration += 1
            logger.info("=== Iteration %d ===", self._iteration)
            self._run_iteration()

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def _run_iteration(self) -> None:
        prompt_mode = self._determine_prompt_mode()
        current_files = self._read_managed_files()

        # 1. Ask LLM for modifications
        proposal = propose_modifications(
            current_files=current_files,
            run_history=self._run_history,
            prompt_mode=prompt_mode,
            model=self.config.model,
        )
        if proposal is None:
            self._consecutive_llm_failures += 1
            logger.warning(
                "LLM returned no valid proposal (failure %d/%d)",
                self._consecutive_llm_failures, _MAX_LLM_FAILURES,
            )
            if self._consecutive_llm_failures >= _MAX_LLM_FAILURES:
                raise RuntimeError(
                    f"LLM failed to produce valid modifications "
                    f"{_MAX_LLM_FAILURES} consecutive times — aborting"
                )
            return
        self._consecutive_llm_failures = 0

        # 2. Write proposed files to working dir (with path safety check)
        self._write_proposed_files(proposal.modified_files)

        # 3. Probe run
        probe_exp = f"agent_iter{self._iteration:04d}_probe_{_ts()}"
        logger.info("Probe run: %s (%d iters)", probe_exp, self.config.probe_iterations)

        probe_result = run_training(
            experiment_name=probe_exp,
            num_envs=self.config.probe_num_envs,
            max_iterations=self.config.probe_iterations,
            device=self.config.device,
            early_stop=True,
        )
        probe_metrics = {
            "final_mean_reward": probe_result.final_mean_reward,
            "iteration_reached": probe_result.iteration_reached,
            "stop_reason": probe_result.stop_reason,
        }

        probe_snapshot = create_snapshot(
            iteration=self._iteration,
            run_type="probe",
            log_dir=probe_result.log_dir,
            metrics=probe_metrics,
            reasoning=proposal.reasoning,
        )

        # 4. Evaluate probe reward curve
        probe_promoted = probe_result.stop_reason == "completed" or (
            probe_result.stop_reason not in ("flatline", "divergence", "error")
            and probe_result.final_mean_reward > 0
        )

        if not probe_promoted:
            logger.info(
                "Probe scrapped: %s (final reward %.3f)",
                probe_result.stop_reason, probe_result.final_mean_reward,
            )
            prune_snapshot(probe_snapshot)
            self._append_history(
                run_type="probe",
                experiment_name=probe_exp,
                stop_reason=probe_result.stop_reason,
                is_promising_flag=False,
                metrics=probe_metrics,
                reasoning=proposal.reasoning,
            )
            return

        # 5. Full run
        full_exp = f"agent_iter{self._iteration:04d}_full_{_ts()}"
        logger.info("Full run: %s (%d iters)", full_exp, self.config.full_iterations)

        full_result = run_training(
            experiment_name=full_exp,
            num_envs=self.config.full_num_envs,
            max_iterations=self.config.full_iterations,
            device=self.config.device,
            early_stop=True,
        )

        # Check if checkpoint exists before eval
        checkpoint_exists = path.exists(
            path.join(full_result.log_dir, "model_*.pt".replace("*", str(full_result.iteration_reached)))
        ) or any(
            True for _ in Path(full_result.log_dir).glob("model_*.pt")
        )

        if not checkpoint_exists:
            logger.warning("No checkpoint found after full run — skipping eval")
            full_snapshot = create_snapshot(
                iteration=self._iteration,
                run_type="full",
                log_dir=full_result.log_dir,
                metrics={"stop_reason": full_result.stop_reason, "error": "no_checkpoint"},
                reasoning=proposal.reasoning,
            )
            prune_snapshot(full_snapshot)
            self._append_history(
                run_type="full",
                experiment_name=full_exp,
                stop_reason="error",
                is_promising_flag=False,
                metrics={"error": "no_checkpoint"},
                reasoning=proposal.reasoning,
            )
            return

        # 6. Eval
        logger.info("Running eval for %s", full_exp)
        try:
            eval_metrics = eval_harness.run_eval(
                log_dir=full_result.log_dir,
                num_steps=250,
                device=self.config.device,
                record_video=True,
            )
        except Exception as exc:
            logger.error("Eval failed: %s", exc)
            eval_metrics = {"error": str(exc), "success": False}

        promising = is_promising(eval_metrics)
        success = eval_metrics.get("success", False)

        full_snapshot = create_snapshot(
            iteration=self._iteration,
            run_type="full",
            log_dir=full_result.log_dir,
            metrics={**eval_metrics, "stop_reason": full_result.stop_reason},
            reasoning=proposal.reasoning,
        )

        if promising:
            logger.info("Full run is PROMISING — keeping snapshot at %s", full_snapshot)
        else:
            prune_snapshot(full_snapshot)

        self._append_history(
            run_type="full",
            experiment_name=full_exp,
            stop_reason=full_result.stop_reason,
            is_promising_flag=promising,
            metrics=eval_metrics,
            reasoning=proposal.reasoning,
        )

        if success:
            logger.info(
                "SUCCESS! Policy achieves clean jump. Snapshot: %s  Video: %s",
                full_snapshot,
                eval_metrics.get("video_path"),
            )
            sys.exit(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _determine_prompt_mode(self) -> Literal["explore", "tune"]:
        if not self._run_history:
            return "explore"
        last = self._run_history[-1]
        # Tune only when the last full run met R9 (is_promising)
        if last.get("run_type") == "full" and last.get("is_promising"):
            return "tune"
        return "explore"

    def _read_managed_files(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in _MANAGED_FILES:
            fpath = path.join(THIS_DIR, name)
            if path.exists(fpath):
                with open(fpath) as f:
                    result[name] = f.read()
        return result

    def _write_proposed_files(self, modified_files: dict[str, str]) -> None:
        for filename, content in modified_files.items():
            target = (_JUMPING_ROOT / filename).resolve()
            # R15: must stay inside spiderbot/jumping/
            if not target.is_relative_to(_JUMPING_ROOT):
                raise ValueError(
                    f"Safety violation: proposed file {filename!r} resolves to "
                    f"{target}, which is outside {_JUMPING_ROOT}"
                )
            target.write_text(content)
            logger.debug("Wrote %s", filename)

    def _backup_managed_files(self) -> None:
        for name in _MANAGED_FILES:
            src = path.join(THIS_DIR, name)
            bak = src + ".bak"
            if path.exists(src) and not path.exists(bak):
                shutil.copy2(src, bak)
                logger.info("Backed up %s → %s", name, name + ".bak")

    def _restore_from_backups(self) -> None:
        for name in _MANAGED_FILES:
            bak = path.join(THIS_DIR, name + ".bak")
            dst = path.join(THIS_DIR, name)
            if path.exists(bak):
                shutil.copy2(bak, dst)
                logger.info("Restored %s from backup", name)

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
        run_type: str,
        experiment_name: str,
        stop_reason: str,
        is_promising_flag: bool,
        metrics: dict,
        reasoning: str,
    ) -> None:
        self._run_history.append({
            "iteration": self._iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "run_type": run_type,
            "experiment_name": experiment_name,
            "stop_reason": stop_reason,
            "is_promising": is_promising_flag,
            "metrics": metrics,
            "reasoning_summary": reasoning[:500] if reasoning else "",
        })
        self._save_run_history()

    def _register_sigint(self) -> None:
        def handler(signum, frame):
            logger.info("SIGINT received — cleaning up and restoring files")
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                    self._current_proc.wait(timeout=10)
                except Exception:
                    pass
            self._save_run_history()
            self._restore_from_backups()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
