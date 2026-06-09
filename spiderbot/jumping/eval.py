"""
Evaluation harness for a trained jumping model.
Loads the saved environment from a training log_dir, runs a short rollout,
and returns jump quality metrics.  Optionally records a video.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import pickle
from os import path

import torch
import genesis as gs
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.wrappers.video import VideoWrapper
from rsl_rl.runners import OnPolicyRunner

THIS_DIR = path.dirname(path.abspath(__file__))

# Contact force threshold below which a link is considered airborne
_AIRBORNE_FORCE_THRESHOLD = 1.0
# Force threshold for "successful" landing (no hard body-to-terrain impact)
_SUCCESS_FORCE_THRESHOLD = 15.0


def run_eval(
    log_dir: str,
    num_envs: int = 1,
    num_steps: int = 250,
    device: str = "gpu",
    record_video: bool = True,
) -> dict:
    """
    Evaluate a trained jumping model.

    Args:
        log_dir: Path to the training output directory (contains cfgs.pkl, model_*.pt, code/).
        num_envs: Number of parallel environments (1 for clean eval).
        num_steps: Rollout length.  250 keeps us inside the 6 s episode limit at 50 Hz.
        device: "gpu" or "cpu".
        record_video: Whether to save an mp4 of the rollout.

    Returns:
        dict with keys:
            max_height_m           – peak CoM z-position during rollout
            height_above_resting_m – peak CoM z above the initial resting height
            forward_distance_m     – max forward (x-axis) displacement from reset position
            max_non_feet_force_N   – peak contact force (N) on non-feet body links
            airborne_steps         – steps with zero foot-to-ground contact
            airborne_fraction      – airborne_steps / num_steps
            success                – bool: good height + positive distance + soft landing
            video_path             – absolute path to saved mp4, or None
    """
    log_dir = path.abspath(log_dir)
    env_file = path.join(log_dir, "code", "environment.py")
    cfgs_file = path.join(log_dir, "cfgs.pkl")

    if not path.exists(env_file):
        raise FileNotFoundError(f"Saved environment not found: {env_file}")
    if not path.exists(cfgs_file):
        raise FileNotFoundError(f"Training config not found: {cfgs_file}")

    # Load the environment class from the saved snapshot
    spec = importlib.util.spec_from_file_location("saved_jumping_env", env_file)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load saved environment from {env_file}: {e}") from e

    EnvClass = mod.SpiderRobotJumpingEnv

    # Load training config (correct pattern — play.py's list-unpack is broken)
    with open(cfgs_file, "rb") as f:
        data = pickle.load(f)
    cfg = data["rsl_rl"]

    # Find latest checkpoint
    checkpoint = _latest_checkpoint(log_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No model checkpoint found in {log_dir}")

    # Genesis init
    backend = gs.cpu if device == "cpu" else gs.gpu
    if device == "cpu":
        torch.set_default_device("cpu")
    seed = data.get("seed", 1)
    gs.init(logging_level="warning", backend=backend, performance_mode=True, seed=seed)

    # Build env
    env = EnvClass(num_envs=num_envs, headless=True)

    video_path: str | None = None
    if record_video:
        video_dir = path.join(log_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = path.join(video_dir, "eval.mp4")
        env = VideoWrapper(
            env,
            camera_attr="camera",
            video_length_sec=num_steps * env.dt + 1.0,
            step_trigger=lambda step: step == 0,
            out_dir=video_dir,
            filename="eval.mp4",
        )

    env = RslRlWrapper(env)
    env.build()

    # Keep a reference to the base env for metric reads
    base_env = env.unwrapped

    # Load policy
    runner = OnPolicyRunner(env, cfg, log_dir, device=gs.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=gs.device)

    # --- Rollout ---
    obs, _ = env.reset()
    initial_pos = base_env.robot.get_pos()[0].clone()  # (3,)

    max_height_m = initial_pos[2].item()
    forward_distance_m = 0.0
    max_non_feet_force_N = 0.0
    airborne_steps = 0

    with torch.no_grad():
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _rew, _done, _info = env.step(actions)

            pos = base_env.robot.get_pos()[0]  # (3,)

            # CoM height
            z = pos[2].item()
            if z > max_height_m:
                max_height_m = z

            # Forward displacement (x-axis)
            dx = (pos[0] - initial_pos[0]).item()
            if dx > forward_distance_m:
                forward_distance_m = dx

            # Non-feet body contact forces
            body_forces = base_env.body_terrain_contact.contacts.norm(dim=-1)  # (n_envs, n_links)
            peak_body_force = body_forces[0].max().item()
            if peak_body_force > max_non_feet_force_N:
                max_non_feet_force_N = peak_body_force

            # Airborne check: any foot contact?
            foot_forces = base_env.foot_contact_manager.contacts.norm(dim=-1)  # (n_envs, n_feet)
            if foot_forces[0].max().item() < _AIRBORNE_FORCE_THRESHOLD:
                airborne_steps += 1

    env.close()

    height_above_resting_m = max_height_m - initial_pos[2].item()

    success = (
        height_above_resting_m > 0.02
        and forward_distance_m > 0.0
        and max_non_feet_force_N < _SUCCESS_FORCE_THRESHOLD
    )

    return {
        "max_height_m": max_height_m,
        "height_above_resting_m": height_above_resting_m,
        "forward_distance_m": forward_distance_m,
        "max_non_feet_force_N": max_non_feet_force_N,
        "airborne_steps": airborne_steps,
        "airborne_fraction": airborne_steps / max(num_steps, 1),
        "success": success,
        "video_path": video_path if (record_video and path.exists(video_path)) else None,
    }


def _latest_checkpoint(log_dir: str) -> str | None:
    checkpoints = glob.glob(path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        return None

    def _iter(p: str) -> int:
        try:
            return int(path.splitext(path.basename(p))[0].split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=_iter)
