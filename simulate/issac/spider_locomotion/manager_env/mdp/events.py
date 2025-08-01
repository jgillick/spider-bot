"""Custom event functions for spider robot training."""

import torch
from typing import Tuple
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


def randomize_joint_stiffness(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    stiffness_range: Tuple[float, float],
    damping_ratio_range: Tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Randomize joint stiffness and damping for domain randomization.
    
    This helps the robot learn to be robust to different actuator properties.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Sample random stiffness and damping values
    num_envs = len(env_ids)
    num_joints = asset.num_joints
    
    # Sample stiffness
    stiffness = sample_uniform(
        stiffness_range[0], stiffness_range[1], (num_envs, num_joints), device=asset.device
    )
    
    # Calculate damping based on damping ratio
    # damping = 2 * damping_ratio * sqrt(stiffness * inertia)
    # For simplicity, we'll use a simplified version
    damping_ratio = sample_uniform(
        damping_ratio_range[0], damping_ratio_range[1], (num_envs, num_joints), device=asset.device
    )
    damping = damping_ratio * torch.sqrt(stiffness) * 0.1  # Simplified damping calculation
    
    # Apply to actuators
    for actuator in asset.actuators.values():
        # Update stiffness
        actuator.stiffness[env_ids] = stiffness[env_ids, actuator.joint_indices]
        # Update damping
        actuator.damping[env_ids] = damping[env_ids, actuator.joint_indices]


def curriculum_base_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    difficulty: float,
    min_height: float,
    max_height: float,
):
    """Curriculum for gradually increasing the target standing height.
    
    Args:
        env: The environment.
        env_ids: Environment indices to apply curriculum to.
        difficulty: Current difficulty level (0 to 1).
        min_height: Minimum target height (easier).
        max_height: Maximum target height (harder).
    """
    # Interpolate target height based on difficulty
    target_height = min_height + (max_height - min_height) * difficulty
    
    # Store in environment for reward calculation
    # This would need to be accessed by the reward function
    if not hasattr(env, "curriculum_data"):
        env.curriculum_data = {}
    
    env.curriculum_data["target_height"] = target_height