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
        # self.rewards.flat_orientation_l2.weight = -5.0
        # self.rewards.dof_torques_l2.weight = -2.5e-5
        # self.rewards.feet_air_time.weight = 0.5
        # self.rewards.undesired_contacts.weight = -0.5

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None
