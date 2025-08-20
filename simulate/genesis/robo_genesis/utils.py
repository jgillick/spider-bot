import torch
from typing import Tuple
import genesis as gs

from genesis.utils.geom import (
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

from .genesis_env import GenesisEnv


def robot_lin_vel(env: GenesisEnv):
    """
    Calculate the robot's linear velocity in the local frame.

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Linear velocity in the local frame
    """
    robot = env.robot
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    return transform_by_quat(robot.get_vel(), inv_base_quat)


def robot_ang_vel(env: GenesisEnv):
    """
    Calculate the robot's angular velocity in the local frame.

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Angular velocity in the local frame
    """
    robot = env.robot
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    return transform_by_quat(robot.get_ang(), inv_base_quat)


def robot_projected_gravity(env: GenesisEnv):
    """
    Calculate the robot's projected gravity in the local frame.

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Projected gravity in the local frame
    """
    robot = env.robot
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    global_gravity = torch.tensor(
        [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
    ).repeat(base_quat.shape[0], 1)
    return transform_by_quat(global_gravity, inv_base_quat)


def robot_relative_quat(
    env: GenesisEnv,
    robot_upright_quat: Tuple[float, float, float, float] = [1.0, 0.0, 0.0, 0.0],
):
    """
    Calculate the robot's quaternion relative to a reference upright orientation.

    Args:
        env: The Genesis environment containing the robot
        robot_upright_quat: Reference quaternion representing upright orientation [w, x, y, z]
                           Defaults to [1.0, 0.0, 0.0, 0.0] (world upright)

    Returns:
        torch.Tensor: Quaternion representing robot's orientation relative to upright reference
                     This can be used to check how much the robot has tipped from upright
    """
    robot = env.robot
    base_quat = robot.get_quat()

    # Convert reference quaternion to tensor and expand to match batch size
    upright_quat = torch.tensor(
        robot_upright_quat, device=base_quat.device, dtype=base_quat.dtype
    )
    upright_quat = upright_quat.expand(base_quat.shape[0], -1)

    # Calculate inverse of the reference upright quaternion
    inv_upright_quat = inv_quat(upright_quat)

    # Transform current quaternion to local frame (relative to upright reference)
    local_quat = transform_quat_by_quat(inv_upright_quat, base_quat)

    return local_quat
