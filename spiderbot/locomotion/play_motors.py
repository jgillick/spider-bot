import os
import glob
import time
import argparse
import torch
import pickle
from serial.tools import list_ports
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.gamepads import Gamepad
from environment import SpiderRobotEnv
from motor.can_motor_controller import CanMotorController
from motor.protocol import AxisState, ControlMode, InputMode

MOTOR_KP = 38.0
MOTOR_KD = 1.2
MOTOR_SMOOTHING = 0.3  # EMA alpha: 0=no movement, 1=no smoothing
MOTOR_DEADBAND = 0.01   # radians — ignore changes smaller than this

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

CAN_BITRATE = 500000

# Map actuator names to CAN node IDs
CAN_ACTUATOR_MAP = {
    "Leg1_Hip": 1,
}


def choose_port():
    """List serial ports and let user select one."""
    IGNORE_PORTS = ["/dev/cu.debug-console", "/dev/cu.Bluetooth-Incoming-Port"]
    ports = list(list_ports.comports())
    ports = [p for p in ports if p.device not in IGNORE_PORTS]
    if not ports:
        print("No serial ports found.")
        port = input("Enter serial port path manually: ").strip()
        return port or None

    print("\nSelect your CAN bus port:")
    for i, p in enumerate(ports, 1):
        desc = p.description or "—"
        print(f"  {i}) {p.device}  —  {desc}")
    print(f"  {len(ports) + 1}) Enter path manually")

    while True:
        try:
            choice = input("\nSelect port [1]: ").strip() or "1"
            n = int(choice)
            if 1 <= n <= len(ports):
                return ports[n - 1].device
            if n == len(ports) + 1:
                return input("Enter port path: ").strip() or None
        except ValueError:
            pass
        print("Invalid choice.")


def connect_to_can() -> CanMotorController:
    """Open the CAN bus, register all motor nodes, and enable closed-loop control."""
    port = choose_port()
    bus = CanMotorController(channel=port, bitrate=CAN_BITRATE)
    for node_id in CAN_ACTUATOR_MAP.values():
        bus.register_node(node_id)
    bus.start()
    for node_id in CAN_ACTUATOR_MAP.values():
        bus.send_clear_errors(node_id)
        time.sleep(0.05)
        bus.send_set_control_mode(node_id, ControlMode.POSITION_CONTROL, InputMode.PASSTHROUGH)
        time.sleep(0.05)
        bus.send_set_state(node_id, AxisState.CLOSED_LOOP_CONTROL)
        time.sleep(0.05)
    return bus


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
    log_path = args.dir
    [cfg] = pickle.load(open(os.path.join(log_path, "cfgs.pkl"), "rb"))
    model = get_latest_model(log_path)
    print(f"Playing model: {model}")

    # Connect to the CAN bus
    print("🚌 Connecting to the CAN bus...")
    bus = connect_to_can()

    print("🔌 Initializing simulator...")

    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Customize environment for playing
    base_env = SpiderRobotEnv(
        num_envs=1, headless=False, terrain="mixed", height_sensor=False, mode="play"
    )
    base_env.build()
    base_env.reward_manager.enabled = False
    base_env.termination_manager.enabled = False
    base_env.self_contact.enabled = False
    base_env.action_manager.noise_scale = 0.0
    base_env.vel_command_manager.range = {
        "lin_vel_x": [-1.0, 1.0],
        "lin_vel_y": [-1.0, 1.0],
        "ang_vel_z": [-0.5, 0.5],
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
    smoothed_positions = {node_id: None for node_id in CAN_ACTUATOR_MAP.values()}
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)

                # Move the actuators
                actuator_values = base_env.action_manager.get_actions_dict()
                for name, node_id in CAN_ACTUATOR_MAP.items():
                    target = actuator_values[name]
                    prev = smoothed_positions[node_id]
                    if prev is None:
                        prev = target
                    smoothed = MOTOR_SMOOTHING * target + (1.0 - MOTOR_SMOOTHING) * prev
                    smoothed_positions[node_id] = smoothed
                    if abs(smoothed - prev) > MOTOR_DEADBAND:
                        bus.send_mit_control(node_id, smoothed, kp=MOTOR_KP, kd=MOTOR_KD)

    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if e.message != "Viewer closed.":
            raise e
    except Exception as e:
        raise e
    finally:
        bus.stop()
    env.close()
    gamepad.stop()


if __name__ == "__main__":
    play()
