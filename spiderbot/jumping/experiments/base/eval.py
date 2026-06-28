"""
Evaluation harness for a trained jumping model.
Runs a short rollout and returns jump quality metrics.  Optionally records a video.
"""

from __future__ import annotations

import glob
import os
import pickle
from os import path

import torch
import genesis as gs
from genesis import constants
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.wrappers.video import VideoWrapper
from rsl_rl.runners import OnPolicyRunner

from .environment import SpiderRobotJumpingEnv
from spiderbot.jumping.config import AIRBORNE_FORCE_THRESHOLD_N, SUCCESS_FORCE_THRESHOLD_N

THIS_DIR = path.dirname(path.abspath(__file__))
DEFAULT_LOG_DIR = path.join(THIS_DIR, "logs", "1_full")


def run_eval(
    log_dir: str = DEFAULT_LOG_DIR,
    num_envs: int = 1,
    num_steps: int = 250,
    device: str = "gpu",
    record_video: bool = True,
) -> dict:
    """
    Evaluate a trained jumping model.

    Args:
        log_dir: Path to the training output directory (contains cfgs.pkl, model_*.pt).
                 Defaults to logs/1_full/ relative to this script.
        num_envs: Number of parallel environments (1 for clean eval).
        num_steps: Rollout length.  250 keeps us inside the 6 s episode limit at 50 Hz.
        device: "gpu" or "cpu".
        record_video: Whether to save an mp4 of the rollout.

    Returns:
        dict with keys:
            max_height_m           - peak CoM z-position during rollout
            height_above_resting_m - peak CoM z above the initial resting height
            forward_distance_m     - max forward (x-axis) displacement from reset position
            max_non_feet_force_N   - peak contact force (N) on non-feet body links
            airborne_steps         - steps with zero foot-to-ground contact
            airborne_fraction      - airborne_steps / num_steps
            success                - bool: good height + positive distance + soft landing
            video_path             - absolute path to saved mp4, or None
    """
    log_dir = path.abspath(log_dir)
    cfgs_file = path.join(log_dir, "cfgs.pkl")

    if not path.exists(cfgs_file):
        raise FileNotFoundError(f"Training config not found: {cfgs_file}")

    # Load training config
    with open(cfgs_file, "rb") as f:
        data = pickle.load(f)
    cfg = data["rsl_rl"]
    seed = data.get("seed", 1)

    # Find latest checkpoint
    checkpoint = _latest_checkpoint(log_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No model checkpoint found in {log_dir}")

    # Genesis init
    backend = constants.backend.cpu if device == "cpu" else constants.backend.gpu
    if device == "cpu":
        torch.set_default_device("cpu")
        
    gs.init(logging_level="warning", backend=backend, performance_mode=True, seed=seed)
    gs_device: str = "cpu"
    if gs.device:
        gs_device = gs.device.type

    # Build env
    base_env = SpiderRobotJumpingEnv(num_envs=num_envs, headless=True)
    env = base_env

    video_path: str | None = None
    if record_video:
        video_dir = path.join(log_dir, '../../')
        video_path = path.join(video_dir, "eval.mp4")
        env = VideoWrapper(
            base_env,
            camera_attr="camera",
            video_length_sec=int(num_steps * env.dt + 1.0),
            step_trigger=lambda step: step == 0,
            out_dir=video_dir,
            filename="eval.mp4",
        )

    env = RslRlWrapper(env)
    env.build()

    # Load policy
    runner = OnPolicyRunner(env, cfg, log_dir, device=gs_device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=gs_device)

    # --- Rollout ---
    obs, _ = env.reset()
    initial_pos = base_env.robot.get_pos()[0].clone()  # (3,)

    max_height_m = initial_pos[2].item()
    forward_distance_m = 0.0
    max_non_feet_force_N = 0.0
    airborne_steps = 0
    early_terminations = 0

    with torch.no_grad():
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _rew, done, info = env.step(actions)

            pos = base_env.robot.get_pos()[0]  # (3,)

            z = pos[2].item()
            if z > max_height_m:
                max_height_m = z

            dx = (pos[0] - initial_pos[0]).item()
            if dx > forward_distance_m:
                forward_distance_m = dx

            body_forces = base_env.body_terrain_contact.contacts.norm(dim=-1)  # (n_envs, n_links)
            peak_body_force = body_forces[0].max().item()
            if peak_body_force > max_non_feet_force_N:
                max_non_feet_force_N = peak_body_force

            foot_forces = base_env.foot_contact_manager.contacts.norm(dim=-1)  # (n_envs, n_feet)
            if foot_forces[0].max().item() < AIRBORNE_FORCE_THRESHOLD_N:
                airborne_steps += 1

            # Count early terminations (body slam, bad orientation, etc.).
            # Timeouts are expected; early terminations mean the robot crashed.
            done_flag = done.any().item() if hasattr(done, "any") else bool(done)
            if done_flag:
                is_timeout = info.get("time_outs", torch.zeros(1)).any().item()
                if not is_timeout:
                    early_terminations += 1

    env.close()

    height_above_resting_m = max_height_m - initial_pos[2].item()
    clean_episode = early_terminations == 0

    success = (
        height_above_resting_m > 0.02
        and forward_distance_m > 0.0
        and max_non_feet_force_N < SUCCESS_FORCE_THRESHOLD_N
        and clean_episode
    )

    return {
        "max_height_m": max_height_m,
        "height_above_resting_m": height_above_resting_m,
        "forward_distance_m": forward_distance_m,
        "max_non_feet_force_N": max_non_feet_force_N,
        "airborne_steps": airborne_steps,
        "airborne_fraction": airborne_steps / max(num_steps, 1),
        "early_terminations": early_terminations,
        "clean_episode": clean_episode,
        "success": success,
        "video_path": video_path if (video_path is not None and path.exists(video_path)) else None,
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


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", nargs="?", default=DEFAULT_LOG_DIR,
                        help="Training log directory (default: logs/1_full/ relative to this script)")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    results = run_eval(
        log_dir=args.log_dir,
        num_steps=args.num_steps,
        device=args.device,
        record_video=not args.no_video,
    )
    print(_json.dumps(results))
