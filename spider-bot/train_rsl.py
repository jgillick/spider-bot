import math
import os
import torch
import shutil
import pickle
import argparse
from datetime import datetime
from importlib import metadata
import genesis as gs
import yaml

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
from environment import SpiderRobotEnv

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install install 'rsl-rl-lib>=2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

DEFAULT_RSL_CONFIG = "./rsl_rl/ppo.yaml"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("-i", "--max_iterations", type=int, default=6000)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-c", "--config", type=str, default=DEFAULT_RSL_CONFIG)
parser.add_argument(
    "-t", "--terrain", type=str, default="flat", help="Set terrain: flat, rough, mixed"
)
parser.add_argument(
    "--lidar",
    action="store_true",
    default=False,
    help="Enable the lidar height sensor.",
)
parser.add_argument("-e", "--experiment_name", type=str)
args = parser.parse_args()


def training_cfg(yaml_path: str, exp_name: str, max_iterations: int, num_envs: int):
    """
    Load the training configuration from the YAML file.
    """

    # Read the YAML file in rsl_rl/ppo.yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Target training batch size of ~98,304 (98,304 / num parallel envs = num_steps_per_env)
    # Based on: https://ar5iv.labs.arxiv.org/html/2109.11978
    num_steps_per_env = math.ceil(98_304 / num_envs)
    cfg["runner"]["num_steps_per_env"] = num_steps_per_env
    cfg["runner"]["experiment_name"] = exp_name
    cfg["runner"]["max_iterations"] = max_iterations

    return cfg["runner"]


def main():
    # Initialize Genesis
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Logging directory
    log_base_dir = "./logs/rsl"
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name += f"_{args.terrain}"
        if args.lidar:
            experiment_name += "_lidar"
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Load training configuration and save snapshot of training configs
    cfg = training_cfg(args.config, experiment_name, args.max_iterations, args.num_envs)
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Create environment
    env = SpiderRobotEnv(
        num_envs=args.num_envs,
        headless=True,
        terrain=args.terrain,
        height_sensor=args.lidar,
    )

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 4 == 0,
    )

    # Build the environment
    env = RslRlWrapper(env)
    env.build()
    env.reset()

    # Train
    print("💪 Training model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.git_status_repos = ["."]
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=False
    )
    env.close()


if __name__ == "__main__":
    main()
