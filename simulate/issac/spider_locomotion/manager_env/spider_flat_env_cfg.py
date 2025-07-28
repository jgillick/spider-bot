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


        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None
