import os
import argparse
import torch
import glob
import pickle
import subprocess
import shutil
from datetime import datetime
from skrl.utils.runner.torch import Runner

import genesis as gs

from genesis_forge.wrappers import (
    SkrlEnvWapper,
    VideoWrapper,
)
from environment import SpiderRobotEnv

SKRL_CONFIG = "./ppo.yaml"

FINAL_VIDEO_DURATION_S = 15

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=3072)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("-d", "--device", type=str, default="gpu")
args = parser.parse_args()


def load_training_config(
    yaml_path: str, max_iterations: int = None, num_envs: int = None
) -> tuple[dict, str]:
    """
    Load the training configuration from the yaml file.

    Args:
        yaml_path: The path to the yaml file.
        max_iterations: The maximum number of iterations.
        log_base_dir: The base directory for the logging directory.

    Returns:
        A tuple containing the training configuration and the logging directory path.
    """
    cfg = Runner.load_cfg_from_yaml(yaml_path)

    # Logging directory
    log_base_dir = "./logs/skrl"
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_base_dir, experiment_name)

    # Update configuration
    cfg["agent"]["experiment"]["directory"] = log_base_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name
    # Target training batch size of ~98,304 (98,304 / num parallel envs = num_steps_per_env)
    # Based on: https://ar5iv.labs.arxiv.org/html/2109.11978
    if num_envs is not None:
        cfg["agent"]["rollouts"] = round(98_304 / num_envs)
    if max_iterations:
        cfg["trainer"]["timesteps"] = max_iterations * cfg["agent"]["rollouts"]

    return cfg, log_path


def get_latest_checkpoint(log_dir: str) -> str:
    """
    Get the latest checkpoint from the log directory
    """
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    # Best checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_agent.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "agent_*.pt"))
    if len(checkpoint_files) == 0:
        print(
            f"Warning: No checkpoint files found at '{checkpoint_dir}' (you might need to train more)."
        )
        return None
    checkpoint_files.sort()
    return checkpoint_files[-1]


def save_env_snapshots(log_dir: str, cfg: dict, files: list[str] = []):
    """
    Save the environment snapshots to the logging directory.

    Args:
        log_dir: The path to the logging directory.
        cfg: The training configuration.
    """
    snapshot_dir = os.path.join(log_dir, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=False)

    # Training config
    pickle.dump([cfg], open(f"{log_dir}/snapshot/cfg.pkl", "wb"))

    # Misc files
    for file in files:
        shutil.copy(file, os.path.join(log_dir, "snapshot", os.path.basename(file)))

    # Git diff
    try:
        result = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, check=True
        )
        diff_file = os.path.join(log_dir, "snapshot/git.patch")
        if result.stdout != "":
            with open(diff_file, "w") as file:
                file.write(result.stdout)
    except:
        pass


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Load training configuration
    cfg, log_path = load_training_config(SKRL_CONFIG, args.max_iterations, args.num_envs)
    print(f"Logging to: {log_path}")
    save_env_snapshots(log_path, cfg, ["./environment.py", SKRL_CONFIG])

    #  Create environment
    env = SpiderRobotEnv(num_envs=args.num_envs, headless=True, terrain="rough")
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )
    env.build()

    # Setup training runner
    env = SkrlEnvWapper(env)
    runner = Runner(env, cfg)

    # Train
    print("💪 Training model...")
    runner.run("train")
    env.close()


if __name__ == "__main__":
    main()
