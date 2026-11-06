"""
Play a trained agent in the simulator while mirroring one leg's actions to real motors.

    python -m spiderbot.locomotion.play_motors ./logs/<TRAINING DIR>

On first run (or with --position), the leg is calibrated using the alignment tool, and
the motor positions are saved to default_pos.json (next to this script). These are the motors' virtual
"zero" positions, which are added to the simulated joint targets.
"""

import os
import glob
import argparse
import torch
import pickle
import sdl2
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.gamepads import Gamepad
from spiderbot.locomotion.environment import SpiderRobotEnv
from spiderbot.motor.can_motor_controller import CanMotorController
from spiderbot.motor.protocol import AxisState
from spiderbot.locomotion.leg_motors import (
    CAN_ACTUATOR_MAP,
    CAN_ACTUATOR_DIRECTION,
    DEFAULT_POS_FILE,
    connect_to_can,
    enable_motors,
    load_default_positions,
    calibrate_default_positions,
)

MOTOR_KP = 25.0
MOTOR_KD = 1.0
MOTOR_SMOOTHING = 0.3  # EMA alpha: 0=no movement, 1=no smoothing
MOTOR_DEADBAND = 0.01  # radians — ignore changes smaller than this

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="gpu",
    help="Where to process the simulation (cpu or gpu)",
)
parser.add_argument(
    "-p",
    "--position",
    action="store_true",
    help="Force recalibration of the actuator zero positions with the alignment tool.",
)
parser.add_argument("dir", help="The training output directory.")
args = parser.parse_args()


def get_default_positions(bus: CanMotorController) -> dict[str, float]:
    """Get the motor zero positions from the saved file, or by calibrating."""
    if args.position:
        return calibrate_default_positions(bus)
    default_pos = load_default_positions()
    if default_pos is not None:
        print(f"Using zero positions from {DEFAULT_POS_FILE}")
        return default_pos
    return calibrate_default_positions(bus)


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

    # Connect to the CAN bus
    print("🚌 Connecting to the CAN bus...")
    bus = connect_to_can()

    try:
        # Motors are left idle during calibration, so the leg can be moved by hand
        default_pos = get_default_positions(bus)

        print("🔌 Initializing simulator...")

        # Processor backend (GPU or CPU)
        backend = gs.constants.backend.gpu
        if args.device == "cpu":
            backend = gs.constants.backend.cpu
            torch.set_default_device("cpu")
        gs.init(logging_level="warning", backend=backend, performance_mode=True)

        # Customize environment for playing
        base_env = SpiderRobotEnv(
            num_envs=1, headless=False, terrain="flat", height_sensor=False, mode="play"
        )
        base_env.build()
        base_env.reward_manager.enabled = False
        base_env.termination_manager.enabled = False
        base_env.self_contact.enabled = False
        base_env.vel_command_manager.range = {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-0.5, 0.5),
        }

        # Connect to gamepad
        print("🎮 Connecting to gamepad...")
        gamepad = Gamepad()
        base_env.vel_command_manager.use_gamepad(
            gamepad, lin_vel_y_axis=0, lin_vel_x_axis=1, ang_vel_z_axis=2
        )

        # Eval
        print("Loading environment...")
        env = RslRlWrapper(base_env)
        runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
        runner.load(model)
        policy = runner.get_inference_policy(device=gs.device)

        obs, _ = env.reset()
        # Motor targets are sent relative to the alignment-tool zero
        for name, node_id in CAN_ACTUATOR_MAP.items():
            bus.set_zero_offset(node_id, default_pos[name])
        enable_motors(bus)
        smoothed_positions: dict[str, float | None] = {name: None for name in CAN_ACTUATOR_MAP}
        sent_positions: dict[str, float | None] = {name: None for name in CAN_ACTUATOR_MAP}
        try:
            with torch.no_grad():
                while True:
                    actions = policy(obs)
                    obs, _rews, _dones, _infos = env.step(actions)

                    # Move the actuators (the bus adds each motor's zero offset)
                    actuator_values = base_env.action_manager.get_actions_dict()
                    for name, node_id in CAN_ACTUATOR_MAP.items():
                        target = actuator_values[name] * CAN_ACTUATOR_DIRECTION[name]
                        prev = smoothed_positions[name]
                        if prev is None:
                            prev = target
                        smoothed = MOTOR_SMOOTHING * target + (1.0 - MOTOR_SMOOTHING) * prev
                        smoothed_positions[name] = smoothed
                        last_sent = sent_positions[name]
                        if last_sent is None or abs(smoothed - last_sent) > MOTOR_DEADBAND:
                            bus.send_mit_control(node_id, smoothed, kp=MOTOR_KP, kd=MOTOR_KD)
                            sent_positions[name] = smoothed

        except KeyboardInterrupt:
            pass
        except gs.GenesisException as e:
            if str(e) != "Viewer closed.":
                raise e
        finally:
            gamepad.stop()
            env.close()
    finally:
        for node_id in CAN_ACTUATOR_MAP.values():
            bus.send_set_state(node_id, AxisState.IDLE)
        bus.stop()


if __name__ == "__main__":
    play()
