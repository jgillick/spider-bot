#!/usr/bin/env python3
"""
Drop the SpiderBot from random joint configurations and record settled ground poses.

Each trial samples random joint positions, holds them with the PD controllers while the
robot settles on the ground, and records the resting joints, pos, and quat.
Poses where any leg is collapsed under the body (the condition penalized by
`FootAngleMdp.reward`), the body is too tilted, or the legs collide with each other are
discarded.

Writes ground_positions.py, which is used by the `random_ground_pose` reset function.

Usage:
    python -m spiderbot.locomotion.generate_random_ground_pos
"""

import os
import re
import math
import argparse
from datetime import datetime, timezone

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat
from genesis_forge.utils import links_by_name_pattern

from spiderbot.environment import SPIDER_XML
from spiderbot.mdp.foot_angle import calculate_leg_angles

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DT = 1 / 50
SUBSTEPS = 4
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

# Actuators (match the nominal values in the locomotion environment)
KP = 30.0
KV = 2.0
MAX_FORCE = 10.0

# Drop / settle tuning
DROP_MARGIN = 0.02  # m clearance between AABB bottom and ground before release
MAX_FALL_STEPS = 300  # max sim steps per trial (~6 s)
STABLE_STEPS = 25  # consecutive stable steps required to record a pose
LIN_THRESH = 0.05  # base linear speed (m/s)
ANG_THRESH = 0.15  # base angular speed (rad/s)
DOF_THRESH = 0.05  # max leg joint speed (rad/s)
LIMIT_SCALE = 0.95  # fraction of the joint range when sampling the pre-drop pose
MAX_SAMPLE_ATTEMPTS = 100  # attempts to sample a valid pre-drop pose

# Hips are sampled around their default positions (from the locomotion environment),
# since sampling the full range mostly results in neighboring legs intersecting.
HIP_DEFAULTS = {
    "R1_Hip": 0.9,
    "R2_Hip": 0.2,
    "R3_Hip": -0.2,
    "R4_Hip": -0.9,
    "L1_Hip": -0.9,
    "L2_Hip": -0.2,
    "L3_Hip": 0.2,
    "L4_Hip": 0.9,
}
HIP_RANGE = 0.35  # +/- rad around the default hip position

# Pose filters
MIN_LEG_ANGLE = 0.0  # same as the FootAngleMdp.reward target_angle
MAX_TILT_DEG = 45.0  # max angle between the body's up-axis and the world's up-axis
SELF_CONTACT_THRESHOLD = 2.0  # same as the self_contact termination threshold

DEFAULT_ITERATIONS = 100
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "ground_positions.py")

JOINT_PATTERN = r"^[RL][1-4]_(Hip|Femur|Tibia)$"
FOOT_PATTERN = "[RL][1-4]_Foot"
# Same links as the self_contact manager in the locomotion environment
SELF_CONTACT_PATTERNS = [
    "[RL][1-4]_Femur",
    "[RL][1-4]_Tibia",
    "[RL][1-4]_Foot",
    ".*_Motor",
]


def collect_leg_dofs(robot) -> tuple[list[str], list[int]]:
    names: list[str] = []
    idx: list[int] = []
    for joint in robot.joints:
        if joint.type != gs.JOINT_TYPE.REVOLUTE:
            continue
        if re.match(JOINT_PATTERN, joint.name):
            names.append(joint.name)
            idx.extend(joint.dofs_idx_local)
    if len(names) != 24:
        raise RuntimeError(f"Expected 24 leg DOFs, found {len(names)}: {names}")
    return names, idx


def setup_actuators(robot, dofs_idx: list[int]) -> None:
    n = len(dofs_idx)
    robot.set_dofs_kp(torch.full((n,), KP, device=gs.device), dofs_idx)
    robot.set_dofs_kv(torch.full((n,), KV, device=gs.device), dofs_idx)
    robot.set_dofs_force_range(
        torch.full((n,), -MAX_FORCE, device=gs.device),
        torch.full((n,), MAX_FORCE, device=gs.device),
        dofs_idx,
    )


def joint_sample_range(
    robot, dof_names: list[str], dofs_idx: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the (lower, upper) range to sample each joint position from.
    """
    lower, upper = robot.get_dofs_limit(dofs_idx)
    if lower.dim() == 2:
        lower, upper = lower[0], upper[0]
    span = (upper - lower) * LIMIT_SCALE
    mid = (upper + lower) * 0.5
    lower, upper = mid - span / 2, mid + span / 2

    for i, name in enumerate(dof_names):
        if name in HIP_DEFAULTS:
            lower[i] = max(lower[i].item(), HIP_DEFAULTS[name] - HIP_RANGE)
            upper[i] = min(upper[i].item(), HIP_DEFAULTS[name] + HIP_RANGE)
    return lower.unsqueeze(0), upper.unsqueeze(0)


def sample_joint_positions(sample_range: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    lower, upper = sample_range
    return lower + torch.rand_like(lower) * (upper - lower)


def legs_collapsed(robot, foot_links_idx: list[int]) -> bool:
    quats = robot.get_links_quat(links_idx_local=foot_links_idx)
    return bool((calculate_leg_angles(quats) < MIN_LEG_ANGLE).any().item())


def too_tilted(robot) -> bool:
    up = torch.tensor([[0.0, 0.0, 1.0]], device=gs.device)
    body_up = transform_by_quat(up, robot.get_quat())
    return body_up[0, 2].item() < math.cos(math.radians(MAX_TILT_DEG))


def has_self_contact(robot, self_contact_links: torch.Tensor) -> bool:
    """
    Check for contact between the legs, from the most recent scene step.
    """
    contacts = robot.get_contacts(with_entity=robot)
    link_a, link_b = contacts["link_a"][0], contacts["link_b"][0]
    tracked = torch.isin(link_a, self_contact_links) & torch.isin(
        link_b, self_contact_links
    )
    force = torch.linalg.norm(contacts["force_a"][0], dim=-1)[tracked]
    return force.numel() > 0 and force.max().item() > SELF_CONTACT_THRESHOLD


def place_above_ground(robot, dofs_idx: list[int], joint_pos: torch.Tensor) -> None:
    robot.set_dofs_position(position=joint_pos, dofs_idx_local=dofs_idx, envs_idx=[0])
    robot.control_dofs_position(joint_pos, dofs_idx, envs_idx=[0])
    pos = torch.tensor([[0.0, 0.0, 0.3]], device=gs.device, dtype=gs.tc_float)
    quat = torch.tensor([INITIAL_QUAT], device=gs.device, dtype=gs.tc_float)
    robot.set_pos(pos, envs_idx=[0], zero_velocity=True)
    robot.set_quat(quat, envs_idx=[0], zero_velocity=True)
    robot.zero_all_dofs_velocity([0])

    # Shift the base so the lowest point of the robot is DROP_MARGIN above the ground
    aabb = robot.get_AABB()
    if aabb.ndim == 3:
        aabb = aabb[0]
    pos[0, 2] += DROP_MARGIN - aabb[0, 2].item()
    robot.set_pos(pos, envs_idx=[0], zero_velocity=True)


def sample_valid_pose(
    scene: gs.Scene,
    robot,
    dofs_idx: list[int],
    sample_range: tuple[torch.Tensor, torch.Tensor],
    foot_links_idx: list[int],
    self_contact_links: torch.Tensor,
) -> torch.Tensor | None:
    """
    Sample random joint positions, rejecting any with legs collapsed under the body
    or intersecting each other.
    """
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        joint_pos = sample_joint_positions(sample_range)
        place_above_ground(robot, dofs_idx, joint_pos)
        if legs_collapsed(robot, foot_links_idx):
            continue
        # Contacts are only detected during a step (the robot is still in the air here)
        scene.step()
        if not has_self_contact(robot, self_contact_links):
            return joint_pos
    return None


def is_stable(robot, dofs_idx: list[int]) -> bool:
    lin = torch.linalg.norm(robot.get_vel()[0]).item()
    ang = torch.linalg.norm(robot.get_ang()[0]).item()
    max_dof = torch.max(torch.abs(robot.get_dofs_velocity(dofs_idx)[0])).item()
    return lin < LIN_THRESH and ang < ANG_THRESH and max_dof < DOF_THRESH


def run_drop_until_stable(scene: gs.Scene, robot, dofs_idx: list[int]) -> bool:
    consecutive = 0
    for _ in range(MAX_FALL_STEPS):
        scene.step()
        if is_stable(robot, dofs_idx):
            consecutive += 1
            if consecutive >= STABLE_STEPS:
                return True
        else:
            consecutive = 0
    return False


def record_pose(robot, dof_names: list[str], dofs_idx: list[int]) -> dict:
    positions = robot.get_dofs_position(dofs_idx)[0].cpu().tolist()
    return {
        "joints": dict(zip(dof_names, positions, strict=True)),
        "pos": robot.get_pos()[0].cpu().tolist(),
        "quat": robot.get_quat()[0].cpu().tolist(),
    }


def write_pose_list(path: str, poses: list[dict], iterations: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        '"""Auto-generated resting poses for SpiderBot locomotion resets. Do not edit by hand."""',
        "",
        f"# Generated: {ts}",
        f"# Requested iterations: {iterations}",
        f"# Recorded poses: {len(poses)}",
        "",
        "RANDOM_GROUND_POSES = [",
    ]
    for pose in poses:
        lines.append("    {")
        lines.append(f"        \"pos\": {pose['pos']!r},")
        lines.append(f"        \"quat\": {pose['quat']!r},")
        lines.append(f"        \"joints\": {pose['joints']!r},")
        lines.append("    },")
    lines.append("]")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def build_scene(headless: bool) -> tuple[gs.Scene, object]:
    scene = gs.Scene(
        show_viewer=not headless,
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-1.0, -0.6, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=True,
            enable_neutral_collision=True,
            multiplier_collision_broad_phase=32,
            max_collision_pairs=40,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(file=SPIDER_XML, pos=[0.0, 0.0, 0.3], quat=INITIAL_QUAT),
    )
    scene.build(n_envs=1)
    return scene, robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop the SpiderBot from random joint poses and record settled ground states "
            "(joints, pos, quat) to a Python module for locomotion training resets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of drop trials to run. Rejected trials are not counted toward the output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to the generated Python module.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Open the Genesis viewer to watch each drop (default: headless).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Genesis compute backend.",
    )
    args = parser.parse_args()

    backend = gs.gpu if args.device == "gpu" else gs.cpu
    if args.device == "cpu":
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    scene, robot = build_scene(headless=not args.show_viewer)
    dof_names, dofs_idx = collect_leg_dofs(robot)
    foot_links_idx = [
        link.idx_local for link in links_by_name_pattern(robot, FOOT_PATTERN)
    ]
    self_contact_links = torch.tensor(
        [
            link.idx
            for pattern in SELF_CONTACT_PATTERNS
            for link in links_by_name_pattern(robot, pattern)
        ],
        device=gs.device,
    )
    sample_range = joint_sample_range(robot, dof_names, dofs_idx)
    setup_actuators(robot, dofs_idx)

    poses: list[dict] = []
    skipped = 0
    for i in range(args.iterations):
        prefix = f"[{i + 1}/{args.iterations}]"
        joint_pos = sample_valid_pose(
            scene, robot, dofs_idx, sample_range, foot_links_idx, self_contact_links
        )
        if joint_pos is None:
            skipped += 1
            print(f"{prefix} skipped (could not sample a valid pre-drop pose)")
            continue
        if not run_drop_until_stable(scene, robot, dofs_idx):
            skipped += 1
            print(f"{prefix} skipped (did not stabilize in time)")
            continue
        if legs_collapsed(robot, foot_links_idx):
            skipped += 1
            print(f"{prefix} skipped (legs collapsed after settling)")
            continue
        if too_tilted(robot):
            skipped += 1
            print(f"{prefix} skipped (body too tilted)")
            continue
        if has_self_contact(robot, self_contact_links):
            skipped += 1
            print(f"{prefix} skipped (self contact)")
            continue

        pose = record_pose(robot, dof_names, dofs_idx)
        poses.append(pose)
        print(f"{prefix} recorded  z={pose['pos'][2]:.4f}")

    write_pose_list(args.output, poses, args.iterations)
    print(f"\nWrote {len(poses)} poses to {os.path.abspath(args.output)} ({skipped} skipped)")


if __name__ == "__main__":
    main()
