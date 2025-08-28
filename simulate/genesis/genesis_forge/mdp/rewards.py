import torch
from typing import Union
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    VelocityCommandManager,
    DofPositionActionManager,
    ContactManager,
)
from genesis_forge.utils import robot_lin_vel, robot_ang_vel, robot_projected_gravity


def base_height(env: GenesisEnv, target_height: Union[float, torch.Tensor]):
    """
    Penalize base height away from target

    Args:
        env: The Genesis environment containing the robot
        target_height: The target height to penalize the base height away from

    Returns:
        torch.Tensor: Penalty for base height away from target
    """
    base_pos = env.robot.get_pos()
    return torch.square(base_pos[:, 2] - target_height)


def dof_similar_to_default(
    env: GenesisEnv,
    dof_action_manager: DofPositionActionManager,
):
    """
    Penalize joint poses far away from default pose

    Args:
        env: The Genesis environment containing the robot
        dof_action_manager: The DOF action manager

    Returns:
        torch.Tensor: Penalty for joint poses far away from default pose
    """
    dof_pos = dof_action_manager.get_dofs_position()
    default_pos = dof_action_manager.default_dofs_pos
    return torch.sum(torch.abs(dof_pos - default_pos), dim=1)


def lin_vel_z(env: GenesisEnv):
    """
    Penalize z axis base linear velocity

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for z axis base linear velocity
    """
    linear_vel = robot_lin_vel(env)
    return torch.square(linear_vel[:, 2])


def action_rate(env: GenesisEnv):
    """
    Penalize changes in actions

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for changes in actions
    """
    actions = env.actions
    last_actions = env.last_actions
    return torch.sum(torch.square(last_actions - actions), dim=1)


def command_tracking_lin_vel(
    env: GenesisEnv,
    vel_cmd_manager: VelocityCommandManager,
    sensitivity: float = 0.25,
):
    """
    Penalize not tracking commanded linear velocity (xy axes)

    Args:
        env: The Genesis environment containing the robot
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error

    Returns:
        torch.Tensor: Penalty for tracking of linear velocity commands (xy axes)
    """
    command = vel_cmd_manager.command
    linear_vel_local = robot_lin_vel(env)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - linear_vel_local[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / sensitivity)


def command_tracking_ang_vel(
    env: GenesisEnv,
    vel_cmd_manager: VelocityCommandManager,
    sensitivity: float = 0.25,
):
    """
    Penalize not tracking commanded angular velocity (yaw)

    Args:
        env: The Genesis environment containing the robot
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error

    Returns:
        torch.Tensor: Penalty for tracking of angular velocity commands (yaw)
    """
    command = vel_cmd_manager.command
    angular_vel = robot_ang_vel(env)
    ang_vel_error = torch.square(command[:, 2] - angular_vel[:, 2])
    return torch.exp(-ang_vel_error / sensitivity)


def has_contact(_env: GenesisEnv, contact_manager: ContactManager, threshold=1.0):
    """
    Returns:
        1 for each contact meeting the threshold
    """
    has_contact = contact_manager.contacts[:, :].norm(dim=-1) > threshold
    return has_contact.sum(dim=1).float()


def flat_orientation_l2(env: GenesisEnv):
    """
    Penalize non-flat base orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for non-flat base orientation
    """
    # Get the projected gravity vector in the robot's base frame
    # This represents how "tilted" the robot is from upright
    projected_gravity = robot_projected_gravity(env)

    # Penalize the xy-components (horizontal tilt) using L2 squared kernel
    # A flat orientation means these components should be close to zero
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
