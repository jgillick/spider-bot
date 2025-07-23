"""Flat terrain configuration for spider locomotion environment."""

from isaaclab.utils import configclass

# Import the base configuration
from .spider_env_cfg import SpiderLocomotionEnvCfg


@configclass
class SpiderLocomotionFlatEnvCfg(SpiderLocomotionEnvCfg):
    """Configuration for spider locomotion on flat terrain"""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
    
        # override rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.is_alive.weight = 0.0

        # override penalties
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.no_contact.weight = -0.5
        self.rewards.termination_penalty.weight = -100.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.base_height_l2.weight = -0.2

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None
