"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Literal, TypedDict
import genesis as gs

from genesis_forge import ManagedEnvironment, GenesisEnv
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    PositionWithinLimitsActionManager,
    ContactManager,
    TerrainManager,
    EntityManager,
    ObservationManager,
    VelocityCommandManager,
)
from genesis_forge.managers.actuator import ActuatorManager, NoisyValue
from genesis_forge.mdp import reset, rewards, terminations, observations

from foot_angle_mdp import FootAngleMdp
from gait_reward import GaitReward


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))
CURRICULUM_CHECK_EVERY_STEPS = 800
CURRICULUM_AVG_SAMPLES = 5

Terrain = Literal["flat", "rough", "mixed"]
EnvMode = Literal["train", "eval", "play"]


class IncConfig(TypedDict):
    inc: float
    limit: float | None


class SpiderRobotEnv(ManagedEnvironment):
    """
    SpiderBot environment for Genesis
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int | None = 6,
        headless: bool = True,
        mode: EnvMode = "train",
        terrain: Terrain = "flat",
        height_sensor: bool = False,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            # max_episode_random_scaling=0.1,
        )
        self.headless = headless
        self.mode = mode
        self.use_height_sensor = height_sensor
        self.curriculum_level = 1
        self.curriculum_samples = []
        self.next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS
        self.terrain_type = terrain

        self.max_velocity_x = 1.2
        self.max_velocity_y = 0.8
        self.max_velocity_z = 1.0
        self.velocity_inc = 0.05
        if terrain != "flat":
            self.velocity_inc = 0.025

        self.construct_scene(terrain)

    """
    Operations
    """

    def construct_scene(self, terrain_type: Terrain) -> gs.Scene:
        """
        Construct the environment scene.
        """
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 2.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
                max_collision_pairs=35,
            ),
        )

        # Create terrain
        checker_image = np.array(Image.open("./assets/checker.png"))
        tiled_image = np.tile(checker_image, (24, 24, 1))
        if terrain_type == "flat":
            self.terrain = self.scene.add_entity(gs.morphs.Plane())
        elif terrain_type == "rough":
            self.terrain = self.scene.add_entity(
                surface=gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(
                        image_array=tiled_image,
                    )
                ),
                morph=gs.morphs.Terrain(
                    pos=(-12, -12, 0),
                    n_subterrains=(1, 1),
                    subterrain_size=(24, 24),
                    vertical_scale=0.001,
                    subterrain_types=[["random_uniform_terrain"]],
                    subterrain_parameters={
                        "random_uniform_terrain": {
                            "min_height": 0.0,
                            "max_height": 0.08,
                            "step": 0.04,
                            "downsampled_scale": 0.25,
                        },
                    },
                ),
            )
        elif terrain_type == "mixed":
            self.terrain = self.scene.add_entity(
                surface=gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(
                        image_array=tiled_image,
                    )
                ),
                morph=gs.morphs.Terrain(
                    n_subterrains=(1, 2),
                    subterrain_size=(24, 12),
                    subterrain_types=[
                        [
                            "flat_terrain",
                            # "discrete_obstacles_terrain",
                            # "pyramid_stairs_terrain",
                            "random_uniform_terrain",
                        ],
                    ],
                    subterrain_parameters={
                        "random_uniform_terrain": {
                            "min_height": 0.0,
                            "max_height": 0.08,
                            "step": 0.04,
                            "downsampled_scale": 0.25,
                        },
                    },
                ),
            )

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=[0.0, 0.0, 0.14],
                quat=[1.0, 0.0, 0.0, 0.0],
            ),
        )

        # Height sensor
        self.height_sensor = None
        if self.use_height_sensor:
            self.height_sensor = self.scene.add_sensor(
                gs.sensors.Lidar(
                    pattern=gs.sensors.GridPattern(resolution=0.2, size=(0.8, 0.8)),
                    entity_idx=self.robot.idx,
                    pos_offset=(0.24, 0.0, 0.0),
                    euler_offset=(0.0, 0.0, 0.0),
                )
            )

        # Add camera
        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
            GUI=self.mode == "play",
        )
        self.camera.follow_entity(self.robot, smoothing=0.05)

        return self.scene

    def config(self):
        """
        Configure the environment managers.
        """
        # Foot angle monitor
        self.foot_angle_mdp = FootAngleMdp(self)

        # Terrain
        self.terrain_manager = TerrainManager(self, terrain_attr="terrain")

        ##
        # Robot manager
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.randomize_terrain_position,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                        "height_offset": 0.15,
                        "subterrain": self.curriculum_terrain,
                    },
                },
            },
        )

        ##
        # Actuators and Actions
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=".*",
            default_pos={
                "Leg[1-2]_Hip": -1.0,
                "Leg[3-6]_Hip": 1.0,
                "Leg[7-8]_Hip": -1.0,
                "Leg[1-8]_Femur": 0.5,
                "Leg[1-8]_Tibia": 0.6,
            },
            kp=NoisyValue(52, 5),
            kv=NoisyValue(1.2, 0.1),
            max_force=NoisyValue(8.0, 1.0),
            frictionloss=NoisyValue(0.1, 0.05),
            # armature=1.68e-4,
            damping=NoisyValue(0.4, 0.1),

        )
        self.action_manager = PositionWithinLimitsActionManager(
            self,
            delay_step=1,
            actuator_manager=self.actuator_manager,
        )

        ##
        # Contact managers

        # Foot/step contact manager
        self.foot_contact_manager = ContactManager(
            self,
            link_names=["Leg[1-8]_Tibia_Foot"],
            with_entity_attr="terrain",
            track_air_time=True,
        )

        # Detect self contacts
        self.self_contact = ContactManager(
            self,
            entity_attr="robot",
            link_names=[
                "Leg[1-8]_Femur",
                "Leg[1-8]_Tibia_Leg",
                ".*_Motor",
            ],
            with_entity_attr="robot",
            with_links_names=[
                "Leg[1-8]_Femur",
                "Leg[1-8]_Tibia_Leg",
                ".*_Motor",
            ],
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "size": 0.05,
                "fps": 10,
            },
        )

        ##
        # Command manager
        self.vel_command_manager = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-0.5, 0.5],
                "ang_vel_z": [-1.0, 1.0],
            },
            resample_time_sec=4.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(
            self,
            cfg={
                "gait": {
                    "weight": 0.25,
                    "fn": GaitReward,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "foot_groups": [
                            ["Leg1_Tibia_Foot", "Leg6_Tibia_Foot"],
                            ["Leg2_Tibia_Foot", "Leg5_Tibia_Foot"],
                            ["Leg3_Tibia_Foot", "Leg8_Tibia_Foot"],
                            ["Leg4_Tibia_Foot", "Leg7_Tibia_Foot"],
                        ],
                    },
                },
                "cmd_linear_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.vel_command_manager,
                    },
                },
                "cmd_angular_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.vel_command_manager,
                    },
                },
                "height": {
                    "weight": -100.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.135,
                        "terrain_manager": self.terrain_manager,
                    },
                },
                "similar_to_default": {
                    "weight": -0.05,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "action_manager": self.action_manager,
                    },
                },
                # "action_rate": {
                #     "weight": -5e-05,
                #     "fn": rewards.action_rate_l2,
                # },
                "flat_orientation": {
                    "weight": -1.5,
                    "fn": rewards.flat_orientation_l2,
                },
                "ang_vel_xy_l2": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2,
                },
                "foot_air_time": {
                    "weight": 0.5,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "vel_cmd_manager": self.vel_command_manager,
                        "time_threshold": 0.2,
                        "time_threshold_max": 0.5,
                    },
                },
                "leg_angle": {
                    "weight": -0.02,
                    "fn": self.foot_angle_mdp.reward,
                },
                "self_contact": {
                    "weight": -0.05,
                    "fn": rewards.contact_force,
                    "params": {
                        "contact_manager": self.self_contact,
                    },
                },
            },
        )

        ##
        # Terminations
        self.termination_manager = TerminationManager(
            self,
            term_cfg={
                "timeout": {
                    "time_out": True,
                    "fn": terminations.timeout,
                },
                "out_of_bounds": {
                    "time_out": True,
                    "fn": terminations.out_of_bounds,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                    },
                },
                "bad_orientation": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 70,
                    },
                },
                "self_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "contact_manager": self.self_contact,
                        "threshold": 2.0,
                    },
                },
                "foot_angle": {
                    "fn": self.foot_angle_mdp.terminate,
                    "params": {
                        "angle_threshold": -0.75,
                    },
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            history_len=5,
            cfg={
                "command": {
                    "fn": self.vel_command_manager.observation,
                },
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "noise": 0.01,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                    "noise": 0.01,
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                    "noise": 0.01,
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                    "noise": 0.01,
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.1,
                    "noise": 0.1,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.raw_actions,
                },
            },
        )
        ObservationManager(
            self,
            history_len=5,
            name="critic",
            cfg={
                "air_time_target": {
                    "fn": self.air_time_observation,
                },
                "height_sensor": {
                    "fn": self.height_sensor_observation,
                },
                "foot_contacts": {
                    "fn": observations.contact_force,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
            },
        )

    def build(self):
        super().build()
        self.foot_angle_mdp.build()

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        obs, reward, terminated, truncated, extras = super().step(actions)

        # Log metrics
        extras["episode"]["Metrics / curriculum_level"] = self.curriculum_level
        extras["episode"]["Metrics / max_velocity"] = self.vel_command_manager.range[
            "lin_vel_x"
        ][1]
        extras["episode"]["Metrics / foot_air_time_midpoint"] = (
            self.reward_manager["foot_air_time"].params["time_threshold"]
            + self.reward_manager["foot_air_time"].params["time_threshold_max"]
        ) / 2.0
        extras["episode"]["Metrics / foot_air_time_weight"] = self.reward_manager[
            "foot_air_time"
        ].weight
        extras["episode"]["Metrics / gait_weight"] = self.reward_manager[
            "gait"
        ].weight
        extras["episode"]["Metrics / similar_to_default_weight"] = self.reward_manager[
            "similar_to_default"
        ].weight

        if self.mode == "play":
            self.camera.render()

        # Finish up
        return obs, reward, terminated, truncated, extras

    def reset(self, envs_idx: list[int] | None = None):
        reset = super().reset(envs_idx)
        if envs_idx is not None:
            self.update_curriculum()
        return reset

    def height_sensor_observation(self, env: GenesisEnv) -> torch.Tensor:
        if self.height_sensor is None:
            return torch.tensor([])
        heights = self.height_sensor.read().distances
        # Convert heights tensor shape: (n_envs, 5, 5) -> (n_envs, 25)
        return heights.flatten(start_dim=-2)

    def air_time_observation(self, env: GenesisEnv) -> float:
        """Return the mid point of the current foot air time target range"""
        params = self.reward_manager["foot_air_time"].params
        mid_point = (params["time_threshold"] + params["time_threshold_max"]) / 2.0
        obs = torch.zeros(env.num_envs, 1, device=gs.device)
        obs[:] = mid_point
        return obs

    def update_curriculum(self):
        """
        Check the curriculum
        """
        # Limit how often we check/update the curriculum
        if self.step_count < self.next_curriculum_check_step:
            return

        # Calculate the average linear velocity reward, over the last few episodes
        # This prevents a momentary spike causing a level up
        if len(self.curriculum_samples) < CURRICULUM_AVG_SAMPLES:
            cmd_linear_vel = self.reward_manager.last_episode_mean_reward(
                "cmd_linear_vel", before_weight=True
            )
            self.curriculum_samples.append(cmd_linear_vel)
            if len(self.curriculum_samples) < CURRICULUM_AVG_SAMPLES:
                return
        cmd_linear_avg = sum(self.curriculum_samples) / len(self.curriculum_samples)

        # Level up
        if cmd_linear_avg > 0.8:
            self.curriculum_level += 1
            self.vel_command_manager.increment_range("lin_vel_x", self.velocity_inc, limit=self.max_velocity_x)
            self.vel_command_manager.increment_range("lin_vel_y", self.velocity_inc, limit=self.max_velocity_y)
            self.vel_command_manager.increment_range("ang_vel_z", self.velocity_inc, limit=self.max_velocity_z)

            # Reduce the similar_to_default reward
            self.reward_manager["similar_to_default"].increment_weight(
                0.001, limit=-0.01
            )

            # Increase the gait reward
            self.reward_manager["gait"].increment_weight(0.05, limit=0.5)

            # Increase the foot_air_time target range
            self.reward_manager["foot_air_time"].increment_weight(0.05, limit=1.0)
            self.reward_manager["foot_air_time"].increment_param(
                "time_threshold", 0.05, limit=0.3
            )
            self.reward_manager["foot_air_time"].increment_param(
                "time_threshold_max", 0.05, limit=1.0
            )

        # Reset the curriculum checks
        self.next_curriculum_check_step = self.step_count + CURRICULUM_CHECK_EVERY_STEPS
        self.curriculum_samples = []
    
    def curriculum_terrain(self) -> Terrain:
        """
        Select the terrain type for the environment.
        """
        if self.terrain_type == "mixed" and self.curriculum_level <= 3:
                return "flat_terrain"
        return None
