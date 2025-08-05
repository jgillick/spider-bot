"""Configuration for the spider locomotion environment."""

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    CurriculumTermCfg,
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
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

# Import spider robot configuration
from ..spider_bot_cfg import SpiderBotCfg
from .managers.metric_manager import MetricsTermCfg
from . import mdp


def joints_by_leg():
    """Return a list of joints, ordered by leg.
    For example:
        - Leg1_Hip
        - Leg1_Femur
        - Leg1_Tibia
        - Leg2_Hip
        - Leg2_Femur
        - Leg2_Tibia
        - ...
        - Leg8_Hip
        - Leg8_Femur
    """
    joints = []
    for leg in range(1, 9):
        joints.append(f".*Leg{leg}_Hip")
        joints.append(f".*Leg{leg}_Femur")
        joints.append(f".*Leg{leg}_Tibia")
    return joints

##
# Scene definition
##
@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the spider robot."""

    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robots
    robot = SpiderBotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SpiderBotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16),
            joint_pos={
                # Hip joints
                ".*Leg1_Hip": 1.0,
                ".*Leg2_Hip": 1.0,
                ".*Leg3_Hip": -1.0,
                ".*Leg4_Hip": -1.0,
                ".*Leg5_Hip": -1.0,
                ".*Leg6_Hip": -1.0,
                ".*Leg7_Hip": 1.0,
                ".*Leg8_Hip": 1.0,
                # Femur joints
                ".*_Femur": 0.9,
                # Tibia joints
                ".*_Tibia": -0.9,
            },
        ),
    )

    # Add contact sensors to all parts connected to the root body
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/.*", history_length=3, track_air_time=True
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0),
    #         lin_vel_y=(-1.0, 1.0),
    #         ang_vel_z=(-1.0, 1.0),
    #         heading=(-math.pi, math.pi),
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the spider robot."""

    # Option 1: Standard full control (current)
    joint_pos = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=joints_by_leg(), rescale_to_limits=True
    )

    # Option 2: Progressive symmetric control (uncomment to use)
    # Start with this for learning to stand:
    # symmetric_pos = ProgressiveSymmetricActionCfg(
    #     asset_name="robot",
    #     stage=1,  # Start with 4D: height + base angles
    #     joint_names=[".*"],
    # )

    # Then gradually increase stage as robot masters each level


@configclass
class ObservationsCfg:
    """Observation specifications for the spider robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Base information
        base_lin_vel = ObsTerm(
            func=velocity_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Commands
        # velocity_commands = ObsTerm(
        #     func=velocity_mdp.generated_commands, params={"command_name": "base_velocity"}
        # )

        # Joint information
        joint_pos = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5)
        )

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
            "restitution_range": (0.0, 0.2),
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

    # Terminated - REDUCED penalty to prevent quick termination learning
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)  # was -100.0
    # Still alive - INCREASED to better balance termination
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)  # was 1.0

    height = RewTerm(
        func=mdp.base_height_gaussian,
        weight=5.0,
        params={
            "target_height": 0.17,
            "min_height": 0.1,
        }
    )

    # -- Standing rewards
    # Reward for maintaining upright posture
    # upright = RewTerm(func=upright_posture, weight=5.0, params={"threshold": 0.85})  # Increased from 3.0, lowered threshold
    # Reward for stable standing with multiple feet
    feet_on_ground = RewTerm(
        func=mdp.contact_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Tibia_Foot"]),
            "min_contacts": 4,
            "threshold": 5.0
        }
    )

    # -- Regularization rewards
    # Penalize deviation from flat orientation
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=3.0)
    # Penalize angular velocity in X and Y to discourage tipping/rolling
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)  # was -0.1
    # Penalize large joint torques to encourage energy efficiency
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # Penalize large joint accelerations for smoother motion
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)  # was -2.5e-7
    # Penalize rapid changes in action for smoother control
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    flat = RewTerm(func=mdp.flat_orientation_exp, weight=3.0)
    tilt = RewTerm(func=mdp.ang_vel_xy_exp, weight=1.0)

    # Reward for maintaining equal force on all feet
    balanced_feet = RewTerm(
        func=mdp.balanced_contact_force,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Tibia_Foot"]),
            "threshold": 5.0,
        },
    )

    # Reward for maintaining equal force between left and right feet
    lateral_balance = RewTerm(
        func=mdp.lateral_force_distribution,
        weight=2.0,
        params={
            "sensor1_cfg": SceneEntityCfg("contact_forces", body_names=["Leg[1-4]_Tibia_Foot"]),
            "sensor2_cfg": SceneEntityCfg("contact_forces", body_names=["Leg[5-8]_Tibia_Foot"]),
            "threshold": 5.0,
        },
    )


    # Penalize all feet in the air
    no_contact = RewTerm(
        func=mdp.desired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Tibia_Foot"])
        },
    )

    # Penalize bad touch forces
    bad_touch = RewTerm(
        func=mdp.contact_forces,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_BadTouch"]),
            "threshold": 30.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the spider robot."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the robot is upside down (orientation deviation too large)
    # RELAXED: Increased angle tolerance to give more recovery time
    upside_down = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},  # was 0.7 (~69 degrees instead of ~40 degrees)
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the spider robot."""

    pass
    # terrain_levels = CurriculumTermCfg(func=velocity_mdp.terrain_levels_vel)


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
class SpiderLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the spider locomotion environment."""

    # Scene settings
    scene: SpiderSceneCfg = SpiderSceneCfg(num_envs=600, env_spacing=2.5)  # Reduced from 4096
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    metrics: MetricsCfg = MetricsCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # Run at 50Hz
        self.episode_length_s = 16.0  # Optimal episode length: 16s = 800 steps at 50Hz

        # Simulation settings
        self.sim.dt = 0.005  # 200Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # Viewer settings
        self.viewer: ViewerCfg = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            eye=(2.0, 1.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
        )

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if getattr(self.curriculum, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
