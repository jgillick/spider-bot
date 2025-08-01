import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def desired_contacts(
    env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def upright_posture(env: ManagerBasedRLEnv, threshold: float = 0.95) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    
    Args:
        env: The environment.
        threshold: Cosine similarity threshold for considering the robot upright.
        
    Returns:
        Reward proportional to how upright the robot is.
    """
    # Get the rotation matrix from the robot's base
    quat = env.scene["robot"].data.root_quat_w
    # Convert quaternion to rotation matrix and extract the z-axis (up) vector
    # For a quaternion [w, x, y, z], the rotation matrix's third column (z-axis) is:
    # [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    up_vec_z = 1 - 2 * (x**2 + y**2)
    
    # Reward is proportional to how aligned the robot's up vector is with world up
    # Only give reward when sufficiently upright
    reward = torch.where(up_vec_z > threshold, up_vec_z, torch.zeros_like(up_vec_z))
    return reward


def standing_stability(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 4,
    threshold: float = 1.0
) -> torch.Tensor:
    """Reward for maintaining stable contact with multiple feet.
    
    Args:
        env: The environment.
        sensor_cfg: Configuration for the contact sensor.
        min_contacts: Minimum number of feet that should be in contact.
        threshold: Force threshold for considering contact (N).
        
    Returns:
        Reward for having sufficient feet in contact.
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check which feet have contact (force > threshold)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    has_contact = contact_forces.norm(dim=-1).max(dim=1)[0] > threshold
    
    # Count number of feet in contact
    num_contacts = has_contact.sum(dim=1)
    
    # Reward for having at least min_contacts feet on ground
    reward = (num_contacts >= min_contacts).float()
    return reward


def proper_leg_configuration(
    env: ManagerBasedRLEnv,
    target_femur_angle: float = 0.9,  # ~45 degrees
    target_tibia_angle: float = -0.9,  # ~90 degrees to femur
    angle_tolerance: float = 0.2
) -> torch.Tensor:
    """Reward for maintaining proper leg joint angles.
    
    Directly rewards the configuration that prevents bad touches:
    - Femur at ~45 degrees down
    - Tibia at ~90 degrees to femur
    """
    # Get joint positions
    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names

    # Get the indices of the femur and tibia joints
    femur_joint_indices = [joint_names.index(f"Leg{leg}_Femur") for leg in range(1, 9)]
    tibia_joint_indices = [joint_names.index(f"Leg{leg}_Tibia") for leg in range(1, 9)]

    # Extract femur and tibia positions
    femur_joints = joint_pos[:, femur_joint_indices]
    tibia_joints = joint_pos[:, tibia_joint_indices]
    
    # Calculate deviation from target angles
    femur_error = torch.abs(femur_joints - target_femur_angle)
    tibia_error = torch.abs(tibia_joints - target_tibia_angle)
    
    # Reward based on how close joints are to target (exponential decay)
    femur_reward = torch.exp(-femur_error / angle_tolerance).mean(dim=1)
    tibia_reward = torch.exp(-tibia_error / angle_tolerance).mean(dim=1)
    
    # Combined reward (both must be good)
    total_reward = femur_reward * tibia_reward
    
    return total_reward


def action_symmetry_bonus(
    env: ManagerBasedRLEnv,
    symmetry_threshold: float = 0.2
) -> torch.Tensor:
    """Reward for symmetric actions across legs.
    
    Encourages the robot to move legs in coordinated patterns
    rather than independently.
    """
    # Get the current actions from the action manager
    actions = env.action_manager._action
    
    # Reshape to (num_envs, 8 legs, 3 joints)
    actions_per_leg = actions.view(-1, 8, 3)
    
    # Calculate variance across legs for each joint type
    femur_variance = actions_per_leg[:, :, 1].var(dim=1)
    tibia_variance = actions_per_leg[:, :, 2].var(dim=1)
    
    # Reward low variance (similar actions across legs)
    total_variance = femur_variance + tibia_variance
    symmetry_reward = torch.exp(-total_variance / symmetry_threshold)
    
    return symmetry_reward
