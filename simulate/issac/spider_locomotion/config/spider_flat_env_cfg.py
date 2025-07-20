"""Flat terrain configuration for spider locomotion environment."""

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# Import the base configuration
from .spider_env_cfg import SpiderLocomotionEnvCfg, SpiderSceneCfg, CurriculumCfg


@configclass
class SpiderFlatSceneCfg(SpiderSceneCfg):
    """Configuration for flat terrain scene with spider robot."""

    # Override terrain to be flat
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


@configclass
class SpiderLocomotionFlatEnvCfg(SpiderLocomotionEnvCfg):
    """Configuration for spider locomotion on flat terrain"""

    # Override scene with flat terrain
    scene: SpiderFlatSceneCfg = SpiderFlatSceneCfg(num_envs=1024, env_spacing=2.5)

    # Disable curriculum for flat terrain
    curriculum = CurriculumCfg()
    curriculum.terrain_levels = None

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Adjust rewards for flat terrain (no terrain adaptation needed)
        # Increase velocity tracking importance
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Adjust episode length for flat terrain
        self.episode_length_s = 30.0
