import os
import sys
import signal
import argparse
import pickle
import torch
from datetime import datetime
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch import Runner

import genesis as gs

from skrl_env_wrapper import SkrlEnvWapper
from environment import SpiderRobotEnv

SKRL_CONFIG = "./ppo.yaml"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_envs", type=int, default=600)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("-d", "--device", type=str, default="gpu")
    args = parser.parse_args()

    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load RL configuration from yaml
    cfg = Runner.load_cfg_from_yaml(SKRL_CONFIG)
    if args.max_iterations:
        cfg["trainer"]["timesteps"] = args.max_iterations * cfg["agent"]["rollouts"]

    # Create log directory
    log_base_dir = os.path.join(THIS_DIR, "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_PPO"
    log_path = os.path.join(log_base_dir, experiment_name)
    video_path = os.path.join(log_path, "videos")
    os.makedirs(log_path, exist_ok=False)

    cfg["agent"]["experiment"]["directory"] = log_base_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name

    # Save config to dir
    pickle.dump([cfg], open(f"{log_path}/cfg.pkl", "wb"))

    #  Create environment
    env = SpiderRobotEnv(num_envs=args.num_envs, video=True, video_dir=video_path)
    env = SkrlEnvWapper(env)

    # Setup interrupt handler
    def shutdown(_sig, _frame):
        print("Shutting down...")
        env.close()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Start runner
    try:
        runner = Runner(env, cfg)
        env.set_data_tracker(runner.agent.track_data)
        runner.run("train")
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
