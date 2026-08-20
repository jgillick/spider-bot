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

    from .agent import is_promising, write_evaluation_doc
    write_evaluation_doc(
        iter_dir=iter_dir,
        eval_metrics=results,
        promising=is_promising(results),
        success=results.get("success", False),
        run_label=f"Eval results ({args.run})",
    )
    print(f"\nUpdated: {iter_dir / 'evaluation.md'}")


if __name__ == "__main__":
    main()
