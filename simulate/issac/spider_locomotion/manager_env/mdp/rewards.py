import torch

from isaaclab.assets import RigidObject
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


def flat_orientation_exp(env: ManagerBasedRLEnv, decay_rate: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward for flat base orientation with exponential decay.

    Args:
        env: The environment.
        decay_rate: Controls how quickly the reward decays as the robot deviates from upright.
                   Higher values = faster decay, lower values = slower decay.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor with values in [0, 1], where 1 is perfectly upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get the projected gravity vector in body frame
    # When robot is perfectly upright, gravity should point straight down in body frame
    # So projected_gravity_b should be [0, 0, -1] (negative because gravity points down)
    # We use the x,y components to measure tilt: when upright, these should be close to zero
    projected_gravity_xy = asset.data.projected_gravity_b[:, :2]

    # Calculate the magnitude of tilt (distance from perfectly upright)
    # When perfectly upright: projected_gravity_xy ≈ [0, 0] → tilt_magnitude ≈ 0
    # When horizontal: projected_gravity_xy ≈ [±1, ±1] → tilt_magnitude ≈ 1.4
    tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)

    # Apply exponential decay: reward = exp(-decay_rate * tilt_magnitude)
    # When tilt_magnitude = 0 (perfectly upright): reward = exp(0) = 1
    # When tilt_magnitude > 0 (tilted): reward decreases exponentially
    reward = torch.exp(-decay_rate * tilt_magnitude)

    return reward

def ang_vel_xy_exp(env: ManagerBasedRLEnv, decay_rate: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for maintaining not tilting the base in the xy plane with exponential decay."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    magnitude = torch.norm(ang_vel_xy, dim=1)
    reward = torch.exp(-decay_rate * magnitude)
    return reward

def foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward for maintaining a stable number of feet on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0] > threshold
    )
    if 4 <= contacts <= 6:
        return 1.0
    elif contacts == 3 or contacts == 7:
        return 0.5
    return 0.0

def balanced_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0, min_contacts: int = 3) -> torch.Tensor:
    """Reward for all target contact sensors having a similar force.

    Args:
        env: The environment.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Force threshold for considering contact (N).
        min_contacts: Minimum number of sensors that should be in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_forces_norm = contact_forces.norm(dim=-1)
    contact_forces_avg = contact_forces_norm.mean(dim=1)

    # Check which contacts meet the threshold
    has_contact = contact_forces_avg > threshold
    num_contacts = has_contact.sum(dim=1)

    # Initialize reward tensor
    reward = torch.zeros_like(num_contacts, dtype=torch.float)

    # Only compute reward for environments with sufficient contacts
    valid_envs = num_contacts >= min_contacts
    if valid_envs.any():
        # Filter to only consider contacts that meet threshold
        contact_forces_filtered = contact_forces_norm.clone()
        # Expand has_contact to match the history dimension
        has_contact_expanded = has_contact.unsqueeze(1).expand_as(contact_forces_norm)
        contact_forces_filtered[~has_contact_expanded] = 0.0  # Zero out contacts below threshold

        # Compute statistics only for valid contacts
        contact_forces_mean = contact_forces_filtered.mean(dim=1)
        contact_forces_std = contact_forces_filtered.std(dim=1)

        # Calculate coefficient of variation for scale-invariant measure
        cv = contact_forces_std / (contact_forces_mean + 1e-8)  # Add small epsilon to avoid division by zero

        # Take mean across sensors to get one reward value per environment
        cv_mean = cv.mean(dim=1)

        # Compute reward only for valid environments
        valid_reward = torch.exp(-3.0 * cv_mean[valid_envs])
        reward[valid_envs] = valid_reward

    return reward

def lateral_force_distribution(env: ManagerBasedRLEnv, sensor1_cfg: SceneEntityCfg, sensor2_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward for maintaining a balanced force between two sets of sensors.

    Args:
        env: The environment.
        sensor1_cfg: Configuration for the first contact sensor group.
        sensor2_cfg: Configuration for the second contact sensor group.
        threshold: Force threshold for considering contact (N).
    """
    contact_sensor1: ContactSensor = env.scene.sensors[sensor1_cfg.name]
    contact_sensor2: ContactSensor = env.scene.sensors[sensor2_cfg.name]

    # Get force data for both sensor groups
    contact_forces1 = contact_sensor1.data.net_forces_w_history[:, :, sensor1_cfg.body_ids, :]
    contact_forces2 = contact_sensor2.data.net_forces_w_history[:, :, sensor2_cfg.body_ids, :]

    # Calculate force magnitudes and average over history
    contact_forces1_norm = contact_forces1.norm(dim=-1)
    contact_forces2_norm = contact_forces2.norm(dim=-1)
    contact_forces1_avg = contact_forces1_norm.mean(dim=1)
    contact_forces2_avg = contact_forces2_norm.mean(dim=1)

    # Check which contacts meet threshold for each group
    has_contact1 = contact_forces1_avg > threshold
    has_contact2 = contact_forces2_avg > threshold

    # Sum forces for each group (only counting contacts above threshold)
    total_force1 = (contact_forces1_avg * has_contact1.float()).sum(dim=1)
    total_force2 = (contact_forces2_avg * has_contact2.float()).sum(dim=1)

    # Check if either group has any contacts
    any_contact1 = has_contact1.any(dim=1)
    any_contact2 = has_contact2.any(dim=1)
    
    # Initialize reward tensor
    reward = torch.zeros_like(total_force1, dtype=torch.float)
    
    # Only compute reward for environments where both groups have at least some contact
    valid_envs = any_contact1 & any_contact2
    if valid_envs.any():
        # Calculate coefficient of variation for balanced force distribution
        forces = torch.stack([total_force1[valid_envs], total_force2[valid_envs]], dim=1)  # Shape: (num_valid_envs, 2)
        force_mean = forces.mean(dim=1)
        force_std = forces.std(dim=1)
        cv = force_std / (force_mean + 1e-8)  # Add small epsilon to avoid division by zero

        # Convert to reward (1.0 = perfectly balanced, approaches 0.0 as imbalance increases)
        valid_reward = torch.exp(-3.0 * cv)
        reward[valid_envs] = valid_reward

    return reward

def contact_reward(
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

def base_height_gaussian(
    env: ManagerBasedRLEnv,
    target_height: float = 0.1,
    min_height: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward for maintaining a target height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    height_error = height - target_height

    # Dynamic Gaussian parameter based on acceptable height range
    min_height_error = target_height - min_height
    desired_reward_at_min = 0.1  # 10% of max reward at minimum acceptable height
    gaussian_param = -torch.log(torch.tensor(desired_reward_at_min)) / (min_height_error**2)

    return torch.exp(-gaussian_param * height_error**2)

