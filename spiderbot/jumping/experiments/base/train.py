import math
import os
import torch
import shutil
import pickle
import argparse
from os import path
from datetime import datetime
from importlib import metadata
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis import constants

from genesis_forge.wrappers import RslRlWrapper
from .environment import SpiderRobotJumpingEnv


THIS_DIR = path.dirname(path.abspath(__file__))
DEFAULT_RSL_CONFIG = path.join(THIS_DIR, "ppo.yaml")

# Training parameters
TOTAL_BATCH = 196_608
MIN_STEPS_PER_ENV = 24
TARGET_MINI_BATCH = 24_576


def training_cfg(yaml_path: str, exp_name: str, max_iterations: int, num_envs: int, experiment_name: str):
    """
    Load the training configuration from the YAML file.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["runner"]["experiment_name"] = exp_name
    cfg["runner"]["max_iterations"] = max_iterations
    cfg["runner"]["experiment_name"] = experiment_name

    # Adjust parameters based on the number of environments
    # Target training batch size of ~196,608 (196,608 / num parallel envs = num_steps_per_env)
    # Based on: https://arxiv.org/abs/2109.11978
    num_steps_per_env = math.ceil(TOTAL_BATCH / num_envs)
    total_batch = num_envs * num_steps_per_env
    num_mini_batches = round(total_batch / TARGET_MINI_BATCH)
    num_learning_epochs = 6 - round(math.log2(num_envs / 2048))
    cfg["runner"]["num_steps_per_env"] = max(MIN_STEPS_PER_ENV, num_steps_per_env)
    cfg["runner"]["algorithm"]["num_mini_batches"] = max(4, num_mini_batches)
    cfg["runner"]["algorithm"]["num_learning_epochs"] = max(3, num_learning_epochs)

    return cfg["runner"]


def train_main(
    num_envs: int = 2096,
    max_iterations: int = 2000,
    device: str = "gpu",
    ppo_config: str = DEFAULT_RSL_CONFIG,
    experiment_name: str | None = None,
    height_sensor: bool = False,
):
    """
    Run a training session. Importable entry point — no module-level argparse.
    Still intended to be invoked as a subprocess for GPU process isolation.
    """
    backend = constants.backend.gpu
    if device == "cpu":
        backend = constants.backend.cpu
        torch.set_default_device("cpu")

    # Logging directory
    log_base_dir = path.join(THIS_DIR, "logs")
    exp_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_base_dir, exp_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Load training configuration
    cfg = training_cfg(ppo_config, "", max_iterations, num_envs, exp_name)
    seed = cfg.get("seed", 1)

    # Save snapshot
    pickle.dump(
        {"args": {"num_envs": num_envs, "max_iterations": max_iterations, "device": device}, "seed": seed, "rsl_rl": cfg},
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Initialize genesis
    gs.init(logging_level="warning", backend=backend, performance_mode=True, seed=seed)
    device_name: str = "cpu"
    if gs.device:
        device_name = gs.device.type

    # Create and build environment
    env = SpiderRobotJumpingEnv(
        num_envs=num_envs,
        headless=True,
        height_sensor=height_sensor,
    )
    env = RslRlWrapper(env)
    env.build()
    env.reset()

    # Train
    print("💪 Training model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=device_name)
    runner.learn(
        num_learning_iterations=max_iterations, init_at_random_ep_len=False
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-n", "--num_envs", type=int, default=2096)
    parser.add_argument("-i", "--max_iterations", type=int, default=2_000)
    parser.add_argument("-d", "--device", type=str, default="gpu")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RSL_CONFIG)
    parser.add_argument(
        "--lidar",
        action="store_true",
        default=False,
        help="Enable the lidar height sensor.",
    )
    parser.add_argument("-e", "--experiment_name", type=str)
    args = parser.parse_args()

    train_main(
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        device=args.device,
        ppo_config=args.config,
        experiment_name=args.experiment_name,
        height_sensor=args.lidar,
    )
