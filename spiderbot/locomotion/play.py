"""
Play a trained agent using a gamepad/joystick.
"""

import os
import glob
import argparse
import torch
import pickle
import sdl2

import genesis as gs
from genesis import constants

from rsl_rl.runners import OnPolicyRunner
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.gamepads import Gamepad
from spiderbot.locomotion.environment import SpiderRobotEnv

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


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model from the log directory
    """
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        exit(1)
    # Sort by the file with the highest number
    sorted_models = sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return sorted_models[-1]


def get_training_config(log_dir: str) -> dict:
    """Load the RSL-RL config saved by train.py"""
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        cfgs = pickle.load(f)
    if isinstance(cfgs, dict) and "rsl_rl" in cfgs:
        return cfgs["rsl_rl"]
    if isinstance(cfgs, (list, tuple)):  # Legacy format
        return cfgs[0]
    return cfgs


def gamepad_connected() -> bool:
    """Check if SDL can see at least one game controller."""
    sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_GAMECONTROLLER)
    try:
        return any(
            sdl2.SDL_IsGameController(i) for i in range(sdl2.SDL_NumJoysticks())
        )
    finally:
        sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_GAMECONTROLLER)


def play():
    """Play a trained agent."""

    if not gamepad_connected():
        print("🎮 No gamepad found. Connect a gamepad and try again.")
        exit(1)

    # Load training files
    log_path = args.dir
    cfg = get_training_config(log_path)
    model = get_latest_model(log_path)
    print(f"Playing model: {model}")

    # Processor backend (GPU or CPU)
    backend = constants.backend.gpu
    if args.device == "cpu":
        backend = constants.backend.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Customize environment for playing
    env = SpiderRobotEnv(
        num_envs=1, headless=False, terrain="flat", height_sensor=False, mode="play"
    )
    env.build()
    env.reward_manager.enabled = False
    env.termination_manager.enabled = False
    env.self_contact.enabled = False
    env.vel_command_manager.range = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-0.5, 0.5),
    }
    env.robot_manager.on_reset["random_ground_pose"].fn.probability = 0

    # Connect to gamepad
    print("🎮 Connecting to gamepad...")
    gamepad = Gamepad()
    env.vel_command_manager.use_gamepad(
        gamepad, lin_vel_y_axis=0, lin_vel_x_axis=1, ang_vel_z_axis=2
    )

    # Eval
    print("Loading environment...")
    env = RslRlWrapper(env)
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise e
    finally:
        gamepad.stop()
        env.close()


if __name__ == "__main__":
    play()
