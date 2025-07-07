#!/usr/bin/env python3
"""Train the spider robot in Isaac Lab using RL algorithms."""

import argparse
import os
import sys

# Add the spider locomotion module to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the spider locomotion environment to register it
import spider_locomotion


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train spider robot in Isaac Lab")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--num_envs", type=int, default=4096, help="Number of environments"
    )
    parser.add_argument(
        "--task", type=str, default="Isaac-SpiderLocomotion-v0", help="Task name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_iterations", type=int, default=10000, help="Maximum training iterations"
    )

    args = parser.parse_args()

    # Import Isaac Lab training utilities
    # This needs to be run from within Isaac Lab environment
    try:
        from isaaclab.scripts.reinforcement_learning.rsl_rl import train
    except ImportError:
        print("Error: This script must be run from within the Isaac Lab environment.")
        print("Please use: ./isaaclab.sh -p simulate/issac/train_spider.py")
        sys.exit(1)

    # Set up training arguments
    train_args = [
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.max_iterations),
    ]

    if args.headless:
        train_args.append("--headless")

    # Run training
    train.main(train_args)


if __name__ == "__main__":
    main()
