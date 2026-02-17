import os
import glob
import argparse
import torch
import pickle
from serial.tools import list_ports
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.gamepads import Gamepad
from environment import SpiderRobotEnv

from motor.simple_can import SimpleCanController

from genesis_forge.gamepads import Gamepad

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


def connect_to_can() -> dict[str, SimpleCanController]:
    """Connect to the CAN bus."""
    port = choose_port()
    conn = {}
    for name, node_id in CAN_ACTUATOR_MAP.items():
        conn[name] = SimpleCanController(node_id, port=port, bitrate=CAN_BITRATE)
        conn[name].connect()
    return conn


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
    actuators = connect_to_can()

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
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)

                # Move the actuators
                actuator_values = base_env.action_manager.get_actions_dict()
                for name, actuator in actuators.items():
                    actuator.update()
                    actuator.send_mit_control(actuator_values[name])

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
