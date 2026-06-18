"""
CLI to re-run the eval harness for a specific experiment iteration.

Usage:
    python -m spiderbot.jumping.run_eval iter0001_20260611_125516
    python -m spiderbot.jumping.run_eval iter0001_20260611_125516 --device cpu --no-video

Prints eval metrics as JSON and updates evaluation.md in the experiment directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from os import path
from pathlib import Path

THIS_DIR = path.dirname(path.abspath(__file__))
_EXPERIMENTS_DIR = Path(THIS_DIR) / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Re-run the eval harness for an experiment iteration.",
    )
    parser.add_argument(
        "iter_dir",
        help="Experiment directory name, e.g. iter0001_20260611_125516",
    )
    parser.add_argument(
        "--run",
        default="1_full",
        choices=["0_probe", "1_full"],
        help="Which training run to evaluate (default: 1_full)",
    )
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    iter_dir = _EXPERIMENTS_DIR / args.iter_dir
    if not iter_dir.exists():
        print(f"Error: experiment directory not found: {iter_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = iter_dir / "logs" / args.run
    if not log_dir.exists():
        print(f"Error: log directory not found: {log_dir}", file=sys.stderr)
        print(f"Available runs: {[p.name for p in (iter_dir / 'logs').iterdir()] if (iter_dir / 'logs').exists() else 'none'}", file=sys.stderr)
        sys.exit(1)

    eval_mod = importlib.import_module(f"spiderbot.jumping.experiments.{args.iter_dir}.eval")

    print(f"Running eval on {args.iter_dir}/{args.run} ...")
    results = eval_mod.run_eval(
        log_dir=str(log_dir),
        num_steps=args.num_steps,
        device=args.device,
        record_video=not args.no_video,
    )

    print(json.dumps(results, indent=2))

    _write_evaluation_doc(iter_dir, args.run, results)
    print(f"\nUpdated: {iter_dir / 'evaluation.md'}")


def _write_evaluation_doc(iter_dir: Path, run_name: str, eval_metrics: dict) -> None:
    from .agent import is_promising

    promising = is_promising(eval_metrics)
    success = eval_metrics.get("success", False)

    # Preserve existing content if present, otherwise start fresh
    existing = (iter_dir / "evaluation.md").read_text() if (iter_dir / "evaluation.md").exists() else ""

    lines = [f"# Evaluation — {iter_dir.name}\n\n"]

    # Keep any probe/full run sections from existing doc
    if existing:
        for section in ("## Probe run", "## Full run"):
            start = existing.find(section)
            if start != -1:
                end = existing.find("\n## ", start + 1)
                lines.append(existing[start: end if end != -1 else len(existing)])
                lines.append("\n")

    lines.append(f"## Eval results ({run_name})\n\n")
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


if __name__ == "__main__":
    main()
