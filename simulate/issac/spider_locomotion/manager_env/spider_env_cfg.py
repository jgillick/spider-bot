"""Configuration for the spider locomotion environment."""

import math
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import isaaclab_tasks.manager_based.locomotion.spider_locomotion.mdp as spider_mdp

# Import spider robot configuration
from ..spider_bot_cfg import SpiderBotCfg

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
        init_state=SpiderBotCfg.InitialStateCfg(pos=(0.0, 0.0, 0.135)),
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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the spider robot."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*Leg[1-8]_Tibia", ".*Leg[1-8]_Hip", ".*Leg[1-8]_Femur"],
        # joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the spider robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Base information
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Commands
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # Joint information
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort, noise=Unoise(n_min=-0.1, n_max=0.1)
        # )

        # Actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
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
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )
    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # -- Reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- Interval

    # Randomly push the robot during training
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the spider robot."""

    # -- Task rewards
    # Reward for matching commanded linear velocity in the XY plane
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Reward for matching commanded angular velocity around the Z axis
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Reward for keeping feet in the air for appropriate durations (encourages stepping)
    feet_air_time = RewTerm(
        func=spider_mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia_Foot"),
            "command_name": "base_velocity",
            "threshold_min": 0.5,
            "threshold_max": 2.0,
        },
    )

    # Still alive
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )

    # -- Regularization rewards
    # Penalize vertical (Z) linear velocity to encourage flat movement
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Penalize angular velocity in X and Y to discourage tipping/rolling
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Penalize large joint torques to encourage energy efficiency
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # Penalize large joint accelerations for smoother motion
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Penalize rapid changes in action for smoother control
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalize offsets from the default joint positions when the command is very small.
    # joint_pos_offset_l2 = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.4)

    # Penalize undesired contacts
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_BadTouch"),
    #         "threshold": 1.0,
    #     },
    # )
    # Penalize if none of the desired contacts are present
    no_contact = RewTerm(
        func=spider_mdp.desired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Tibia_Foot"])
        },
    )

    # Terminated
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # Penalize feet sliding
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia_Foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Tibia_Foot"),
        },
    )

    # -- Optional rewards

    # Reward for keeping the robot's body flat (upright orientation)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)

    # Penalize joint positions if they cross the soft limits.
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # Penalize deviation from target base height
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.134}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the spider robot."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the robot is upside down (orientation deviation too large)
    upside_down = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},
    )

    # Terminate if the robot if the "BadTouch" bodies are in contact
    bad_touch = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_BadTouch"]),
            "threshold": 1.0,
        },
    )

    # Terminate if the robot bottoms out
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 ".*_Hip_actuator_assembly_Hip_Bracket",
    #                 ".*_Femur_actuator_assembly_Motor",
    #                 ".*_Knee_actuator_assembly_Motor",
    #             ],
    #         ),
    #         "threshold": 10.0,
    #     },
    # )
    ground_contact = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.09,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the spider robot."""

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class SpiderLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the spider locomotion environment."""

    # Scene settings
    scene: SpiderSceneCfg = SpiderSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # Run at 50Hz
        self.episode_length_s = 20.0  # max episode length in seconds

        # Simulation settings
        self.sim.dt = 0.005  # 200Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

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
