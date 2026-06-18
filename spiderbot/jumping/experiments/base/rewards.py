"""
Custom reward functions for the SpiderRobotJumpingEnv.

Add new reward functions here. Import them by name in environment.py and reference
them in the RewardManager config.

Signature:  def fn(env) -> torch.Tensor   # shape (num_envs,), float
            positive = reward, negative = penalty

Available data:
  env.robot_manager.entity.get_pos()             → (num_envs, 3)   CoM xyz
  env.robot_manager.entity.get_vel()             → (num_envs, 3)   CoM linear velocity xyz
  env.foot_contact_manager.contacts              → (num_envs, 8, 3) foot contact force vectors
  env.foot_contact_manager.contacts.norm(dim=-1) → (num_envs, 8)   per-foot force magnitude
  env.body_terrain_contact.contacts              → contact forces for non-foot body links
  env.action_manager.get_dofs_position()         → (num_envs, num_dofs)
  env.action_manager.get_dofs_velocity()         → (num_envs, num_dofs)
"""
from __future__ import annotations

import torch


def all_feet_airborne(env) -> torch.Tensor:
    """Returns 1.0 for each env where all 8 feet are simultaneously off the ground."""
    force_norms = env.foot_contact_manager.contacts.norm(dim=-1)  # (num_envs, 8)
    return (force_norms < 1.0).all(dim=-1).float()
