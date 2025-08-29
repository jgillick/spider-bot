"""
Play a trained agent using a gamepad/joystick.
By default this uses the Logitech F310 gamepad.
"""

import os
import argparse
import torch
import pickle
from skrl.utils.runner.torch import Runner

import genesis as gs

from genesis_forge.rl.skrl import SkrlEnvWapper
from genesis_forge.rl.skrl.utils import get_latest_checkpoint
from environment import SpiderRobotEnv

from genesis_forge.gamepads.logitech import (
    LogitechGamepad as Gamepad,
    LogitechGamepadProduct,
)

GAMEPAD_PRODUCT = LogitechGamepadProduct.F310

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="gpu",
    help="Where to process the simulation (cpu or gpu)",
)
parser.add_argument("dir", help="The training output directory.")
args = parser.parse_args()


def get_training_config():
    cfg = os.path.join(args.dir, "cfg.pkl")
    if not os.path.exists(cfg):
        cfg = os.path.join(args.dir, "snapshot", "cfg.pkl")
    if not os.path.exists(cfg):
        return None
    with open(cfg, "rb") as f:
        return pickle.load(f)


def play():
    """Play a trained agent."""

    # Load training files
    checkpoint_path = get_latest_checkpoint(args.dir)
    if checkpoint_path is None:
        print(f"ERROR: No training agent checkpoint found in '{args.dir}'.")
        return
    cfg = get_training_config()
    if cfg is None or len(cfg) == 0:
        print(f"ERROR: No training configurations found in '{args.dir}'.")
        return
    cfg = cfg[0]

    # Connect to gamepad
    gamepad = Gamepad(GAMEPAD_PRODUCT)

    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Customize environment for playing
    env = SpiderRobotEnv(num_envs=1, headless=False, mode="play")
    env.build()
    env.reward_manager.enabled = False
    env.termination_manager.enabled = False
    env.bad_touch_contact.enabled = False
    env.action_manager.noise_scale = 0.0
    env.velocity_command.use_gamepad(
        gamepad, lin_vel_y_axis=0, lin_vel_x_axis=1, ang_vel_z_axis=2
    )

    # Setup runner configuration
    cfg["trainer"]["close_environment_at_exit"] = False
    cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    cfg["agent"]["random_timesteps"] = -1

    env = SkrlEnvWapper(env)
    runner = Runner(env, cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    # Play
    states, _infos = env.reset()
    timestep = 0
    try:
        while True:
            timestep += 1
            # Get actions from agent
            (actions, _prob, outputs) = runner.agent.act(
                states, timestep=timestep, timesteps=0
            )
            actions = outputs.get("mean_actions", actions)

            # Perform step
            next_states, _rewards, terminated, truncated, _infos = env.step(actions)
            env.render()

            # Check for termination/truncation
            if terminated.any() or truncated.any():
                states, _infos = env.reset()
                timestep = 0
            else:
                states = next_states
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if e.message != "Viewer closed.":
            raise e
    except Exception as e:
        raise e
    env.close()
    gamepad.stop()


if __name__ == "__main__":
    play()
