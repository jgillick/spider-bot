import os
import glob
import signal
import argparse
import pickle
import torch
from datetime import datetime
from skrl.utils.runner.torch import Runner

import genesis as gs

from genesis_forge import (
    create_skrl_env,
    DataLoggerWrapper,
    VideoWrapper,
    VideoCameraConfig,
    VideoFollowRobotConfig,
)
from environment import SpiderRobotEnv

SKRL_CONFIG = "./ppo.yaml"

CAMERA_CONFIG: VideoCameraConfig = {
    "pos": (-2.5, -1.5, 1.0),
    "lookat": (0.0, 0.0, 0.0),
    "fov": 40,
    "env_idx": 0,
    "debug": True,
}
CAMERA_FOLLOW_CONFIG: VideoFollowRobotConfig = {
    "fixed_axis": (None, None, None),
}

FINAL_VIDEO_DURATION_S = 10

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=600)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("-d", "--device", type=str, default="gpu")
args = parser.parse_args()


def load_training_config(max_iterations: int) -> dict:
    """
    Load the configuration from the yaml file.
    """
    cfg = Runner.load_cfg_from_yaml(SKRL_CONFIG)

    # Logging directory
    log_base_dir = os.path.join(THIS_DIR, "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{cfg['agent']['class']}"
    log_path = os.path.join(log_base_dir, experiment_name)
    os.makedirs(log_path, exist_ok=False)

    # Update configuration
    cfg["agent"]["experiment"]["directory"] = log_base_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name
    if max_iterations:
        cfg["trainer"]["timesteps"] = max_iterations * cfg["agent"]["rollouts"]

    # Save config to logging directory
    pickle.dump([cfg], open(f"{log_path}/cfg.pkl", "wb"))

    return cfg, log_path


def train(
    cfg: dict,
    num_envs: int,
    video_path: str,
):
    """
    Train the agent.
    """
    #  Create environment
    env = SpiderRobotEnv(num_envs=num_envs, headless=True)
    env = DataLoggerWrapper(env)
    env = VideoWrapper(
        env,
        video_length_s=5,
        out_dir=video_path,
        camera=CAMERA_CONFIG,
        follow_robot=CAMERA_FOLLOW_CONFIG,
    )
    env.build()

    # Setup training runner
    skrl_env = create_skrl_env(env)
    runner = Runner(skrl_env, cfg)
    env.set_data_tracker(runner.agent.track_data)

    # Train
    print("💪 Training model...")
    runner.run("train")
    skrl_env.close()


def record_video(cfg: dict, log_path: str, video_path: str):
    """Record a video of the best performing episode."""
    # Recording environment
    env = SpiderRobotEnv(num_envs=1)
    env = VideoWrapper(
        env,
        out_dir=video_path,
        filename="best.mp4",
        video_length_s=FINAL_VIDEO_DURATION_S,
        camera=CAMERA_CONFIG,
        follow_robot=CAMERA_FOLLOW_CONFIG,
    )

    # Update timesteps to only record the final video
    cfg["trainer"]["timesteps"] = env._video_length_steps

    # Load best checkpoint
    # Otherwise load the latest checkpoint, with the highest number
    checkpoint_dir = os.path.join(log_path, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_agent.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "agent_*.pt"))
        if len(checkpoint_files) == 0:
            print(
                f"Warning: No checkpoint files found at '{checkpoint_dir}' (you might need to train more)."
            )
            return
        checkpoint_files.sort()
        checkpoint_path = checkpoint_files[-1]

    # Setup runner
    skrl_env = create_skrl_env(env)
    runner = Runner(skrl_env, cfg)
    runner.agent.load(checkpoint_path)

    # Eval
    print("🎬 Recording video of best model...")
    env.build()
    runner.run("eval")
    skrl_env.close()


def main():
    # Setup interrupt handler
    def shutdown(_sig, _frame):
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Processor backend
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    cfg, log_path = load_training_config(args.max_iterations)
    video_path = os.path.join(log_path, "videos")
    print(f"Logging to: {log_path}")

    # Train agent
    train(cfg, args.num_envs, video_path)

    # Record a video of the final episode
    record_video(cfg, log_path, video_path)


if __name__ == "__main__":
    main()
