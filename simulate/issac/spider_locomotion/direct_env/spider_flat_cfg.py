# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import numpy as np
from gymnasium import spaces

from ..spider_bot_cfg import SpiderBotCfg


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
    episode_length_s = 20.0
    decimation = 4
    observation_space = 60
    state_space = 0
    action_space = spaces.Box(
        low=-1,
        high=1,
        shape=(24,),
        dtype=np.float32,
    )

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
        init_state=SpiderBotCfg.InitialStateCfg(pos=(0.0, 0.0, 0.135))
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Body/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # command limits
    max_command_velocity = 0.3 # m/s

    # reward scales
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 0.5
    feet_air_time_reward_scale = 0.5

    # penalty scales
    z_vel_reward_scale = -0.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = 0.0
    undesired_contact_reward_scale = -0.2
    flat_orientation_reward_scale = -0.0 
