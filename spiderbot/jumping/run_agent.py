"""
CLI entry point for the autonomous jumping agent.
Usage: python -m spiderbot.jumping.run_agent [options]
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from .agent import AgentConfig, JumpingAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_agent",
        description=(
            "Autonomous RL agent that iteratively trains the spider robot to jump. "
            "Requires ANTHROPIC_API_KEY in the environment."
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--probe-iterations",
        type=int,
        default=600,
        metavar="N",
        help="Training iterations per probe run (default: 600)",
    )
    parser.add_argument(
        "--full-iterations",
        type=int,
        default=1500,
        metavar="N",
        help="Training iterations for full runs (default: 1500)",
    )
    parser.add_argument(
        "--probe-envs",
        type=int,
        default=512,
        metavar="N",
        help="Parallel environments during probe runs (default: 512)",
    )
    parser.add_argument(
        "--full-envs",
        type=int,
        default=2096,
        metavar="N",
        help="Parallel environments during full runs (default: 2096)",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Compute device (default: gpu)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing run_history.json (continue from last iteration)",
    )
    args = parser.parse_args()

    # Load .env file before reading the API key (no-op if already set in env)
    load_dotenv()

    # Validate API key before touching anything else
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with:  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    config = AgentConfig(
        model=args.model,
        probe_iterations=args.probe_iterations,
        full_iterations=args.full_iterations,
        probe_num_envs=args.probe_envs,
        full_num_envs=args.full_envs,
        device=args.device,
    )

    # Print resolved config (key is redacted)
    print("Starting autonomous jumping agent with config:")
    print(f"  model:            {config.model}")
    print(f"  probe_iterations: {config.probe_iterations}")
    print(f"  full_iterations:  {config.full_iterations}")
    print(f"  probe_num_envs:   {config.probe_num_envs}")
    print(f"  full_num_envs:    {config.full_num_envs}")
    print(f"  device:           {config.device}")
    print(f"  resume:           {args.resume}")
    print(f"  ANTHROPIC_API_KEY: [set, {len(api_key)} chars]")

    if not args.resume:
        import os as _os
        history_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "run_history.json"
        )
        if _os.path.exists(history_path):
            print(
                f"\nNote: run_history.json exists. "
                f"Pass --resume to continue from iteration {_count_history(history_path) + 1}, "
                f"or it will be ignored (starting from iteration 1)."
            )

    JumpingAgent(config).run()


def _count_history(history_path: str) -> int:
    import json
    try:
        with open(history_path) as f:
            return len(json.load(f))
    except Exception:
        return 0


if __name__ == "__main__":
    main()
