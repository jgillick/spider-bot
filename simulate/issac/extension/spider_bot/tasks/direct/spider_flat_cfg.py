# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import numpy as np
from gymnasium import spaces

from spider_bot.spider_bot_cfg import SpiderBotCfg


@configclass
class RewardsCfg:
    """Reward terms"""

    # Still alive
    # is_alive = RewTerm(
    #     func=mdp.is_alive,
    #     weight=1.0,
    # )

    # -- Regularization rewards
    # Penalize rapid movement along the Z axis (vertical)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Penalize for not maintaining a flat orientation
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=2.5)
    # Penalize angular velocity in X and Y to discourage tipping/rolling
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Penalize large joint torques to encourage energy efficiency
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # Penalize large joint accelerations for smoother motion
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Penalize rapid changes in action for smoother control
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalize feet sliding
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia_Foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_Tibia_Foot"),
    #     },
    # )

    # Penalize deviation from target base height
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.13}
    )

    # Terminated
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.0, 0.2),
            "operation": "add",
        },
    )


@configclass
class SpiderFlatEnvCfg(DirectRLEnvCfg):
    # env
    debug_vis = True
    episode_length_s = 16.0
    decimation = 4
    state_space = 0
    observation_space = 84
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(24,),
    )
    rewards: RewardsCfg = RewardsCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True
    )

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = SpiderBotCfg(
        prim_path="/World/envs/env_.*/Robot",
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Body/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Viwer settings
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=(2.0, 1.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
    )

    # command limits
    max_command_velocity = 0.3  # m/s

    # reward scales
    lin_vel_reward_scale = 4.0  # Increased reward for forward movement
    yaw_rate_reward_scale = 0.5
    feet_air_time_reward_scale = 2.0  # Increased reward for stepping
    feet_on_ground_reward_scale = 5.0  # Reward for each foot in contact with the ground

    # penalty scales
    z_vel_reward_scale = -0.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = 0.0
    undesired_contact_reward_scale = -1.5  # Reduced penalty from -2.0
    flat_orientation_reward_scale = -100.0  # Strong penalty for being tilted (uprightness)
    # Add a new penalty for low height
    low_height_penalty_scale = -5.0  # Reduced penalty for being too low
    low_height_buffer_steps = 10  # Number of consecutive steps below threshold before termination
    bad_touch_penalty_scale = -1.0  # Reduced penalty for continuous BadTouch contact
    standing_height_reward_scale = 5.0  # Much stronger reward for standing up
    curriculum_standing_only = True  # If True, only reward standing, not walking
    low_height_threshold = 0.135  # Raise threshold for low height 
