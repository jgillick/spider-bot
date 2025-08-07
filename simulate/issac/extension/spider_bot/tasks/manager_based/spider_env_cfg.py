"""Configuration for the spider locomotion environment."""

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
import isaaclab.sim as sim_utils

# Import spider robot configuration
from spider_bot.spider_bot_cfg import SpiderBotCfg
from .managers import MetricsTermCfg
from . import mdp


##
# Scene definition
##


@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the spider robot."""

    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            # restitution_combine_mode="multiply",
            # restitution=0.0,
            static_friction=1.0,
            dynamic_friction=1.0,
        # ),
    )

    # Robots
    robot = SpiderBotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SpiderBotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16),
            joint_pos={
                # Hip joints
                "Leg[1-2]_Hip": -1.0,
                "Leg[3-6]_Hip": 1.0,
                "Leg[7-8]_Hip": -1.0,
                # Femur joints
                "Leg[1-8]_Femur": 0.5,
                # Tibia joints
                "Leg[1-8]_Tibia": 0.5,
            },
        ),
    )

    # Add contact sensors to all parts connected to the root body
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/.*", history_length=3, track_air_time=True
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the spider robot."""

    joint_pos = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=".*",
        rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the spider robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Base information
        base_lin_vel = ObsTerm(
            func=velocity_mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(
            func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),  # Reduced from ±0.05
        )

        # Joint information
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for domain randomization and disturbances."""

    # -- Startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Friction of rubber feet
            "static_friction_range": (0.8, 1.5),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # -- Reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the spider robot."""

    # -- Aliveness
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # -- Task
    height = RewTerm(
        func=mdp.base_height_l2,
        weight=10.0,
        params={"target_height": 0.17},
    )
    # Reward for maintaining equal force on all feet
    balanced_feet = RewTerm(
        func=mdp.balanced_contact_force,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_Tibia_Foot"]
            ),
            "threshold": 5.0,
        },
    )

    # -- Penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the spider robot."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the robot is upside down (orientation deviation too large)
    upside_down = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.5},
    )


@configclass
class MetricsCfg:
    """Metrics to average over the episode and send to the logger"""

    height = MetricsTermCfg(
        func=mdp.robot_height,
    )
    bad_touch = MetricsTermCfg(
        func=mdp.max_contact_forces,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_BadTouch"]),
        },
    )
    feet_force = MetricsTermCfg(
        func=mdp.mean_contact_forces,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*Tibia_Foot"]),
        },
    )


##
# Environment configuration
##


@configclass
class SpiderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the spider locomotion environment."""

    # Scene settings
    scene: SpiderSceneCfg = SpiderSceneCfg(num_envs=600, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    metrics: MetricsCfg = MetricsCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # Run at 50Hz
        self.episode_length_s = 16.0

        # Simulation settings
        self.sim.dt = 0.001  # 500Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # Viewer settings
        self.viewer: ViewerCfg = ViewerCfg(
            # origin_type="asset_root",
            # asset_name="robot",
            # env_index=0, 
            # eye=(2.5, 1.5, 1.5),
            # lookat=(0.0, 0.0, 0.0),
            eye=(5.0, -2.0, 3.0), 
        )

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if getattr(self.curriculum, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt
