"""
Custom termination functions for the SpiderRobotJumpingEnv.

Add new termination functions here. Import them by name in environment.py and reference
them in the TerminationManager config.

Signature:  def fn(env) -> torch.Tensor   # shape (num_envs,), bool
            True = terminate (reset) this env

Available data: same as rewards.py (env.robot_manager, env.foot_contact_manager, etc.)
"""
from __future__ import annotations

import torch
