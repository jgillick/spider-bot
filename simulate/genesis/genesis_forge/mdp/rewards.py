import torch
from typing import Sequence, Callable, Union
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import VelocityCommandManager
from genesis_forge.utils import robot_lin_vel, robot_ang_vel


def base_height(env: GenesisEnv, target_height: float):
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
    dof_idx: Union[Callable[[], Sequence[int]], Sequence[int]],
    default_dof_pos: Sequence[float],
):
    """
    Penalize joint poses far away from default pose

    Args:
        env: The Genesis environment containing the robot
        default_dof_pos: The default joint positions

    Returns:
        torch.Tensor: Penalty for joint poses far away from default pose
    """
    robot = env.robot
    if callable(dof_idx):
        dof_idx = dof_idx()
    dof_pos = robot.get_dofs_position(dof_idx)
    return torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)


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
    linear_vel = robot_lin_vel(env)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - linear_vel[:, :2]), dim=1)
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
