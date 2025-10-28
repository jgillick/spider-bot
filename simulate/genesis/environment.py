"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import re
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
    PositionActionManager,
    PositionWithinLimitsActionManager,
    ContactManager,
    TerrainManager,
    EntityManager,
    ObservationManager,
)
from genesis_forge.mdp import reset, rewards, terminations

from foot_angle_mdp import FootAngleMdp
from gait_command import GaitCommandManager


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))
CURRICULUM_CHECK_EVERY_STEPS = 800

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
        max_episode_length_s: int | None = 8,
        headless: bool = True,
        mode: EnvMode = "train",
        terrain: Terrain = "flat",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )
        self.headless = headless
        self.mode = mode
        self._curriculum_level = 1
        self._next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS
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
                            "max_height": 0.1,
                            "step": 0.05,
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
                    n_subterrains=(1, 3),
                    subterrain_size=(12, 12),
                    subterrain_types=[
                        [
                            "flat_terrain",
                            "discrete_obstacles_terrain",
                            "pyramid_stairs_terrain",
                        ],
                    ],
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
                    },
                },
            },
        )

        ##
        # DOF action manager
        self.action_manager = PositionWithinLimitsActionManager(
            self,
            joint_names=".*",
            default_pos={
                # Hip joints
                "Leg[1-2]_Hip": -1.0,
                "Leg[3-6]_Hip": 1.0,
                "Leg[7-8]_Hip": -1.0,
                # Femur joints
                "Leg[1-8]_Femur": 0.5,
                # Tibia joints
                "Leg[1-8]_Tibia": 0.6,
            },
            pd_kp=50,
            pd_kv=1.2,
            max_force=8.0,
            frictionloss=0.1,
            noise_scale=0.02,
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
            },
        )

        ##
        # Gait command manager
        self.gait_command_manager = GaitCommandManager(
            self,
            foot_names={
                "L1": "Leg1_Tibia_Foot",
                "L2": "Leg2_Tibia_Foot",
                "L3": "Leg3_Tibia_Foot",
                "L4": "Leg4_Tibia_Foot",
                "R1": "Leg5_Tibia_Foot",
                "R2": "Leg6_Tibia_Foot",
                "R3": "Leg7_Tibia_Foot",
                "R4": "Leg8_Tibia_Foot",
            },
            velocity_range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-0.5, 0.5],
                "ang_vel_z": [-0.5, 0.5],
            },
            resample_time_sec=4.0,
            jumping_probability=0.0,
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
                # "gait_phase_reward": {
                #     "weight": 1.5,
                #     "fn": self.gait_command_manager.gait_phase_reward,
                #     "params": {
                #         "contact_manager": self.foot_contact_manager,
                #     },
                # },
                "foot_sync": {
                    "weight": 0.25,
                    "fn": self.gait_command_manager.foot_sync_reward,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                # "foot_height_reward": {
                #     "weight": 1.0,
                #     "fn": self.gait_command_manager.foot_height_reward,
                # },
                # "jump": {
                #     "weight": 1.0,
                #     "fn": self.gait_command_manager.jump_reward,
                #     "params": {
                #         "entity_manager": self.robot_manager,
                #         "terrain_manager": self.terrain_manager,
                #     },
                # },
                "cmd_linear_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.gait_command_manager,
                    },
                },
                "cmd_angular_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.gait_command_manager,
                    },
                },
                "height": {
                    "weight": -50.0,
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
                    "weight": -1.0,
                    "fn": rewards.flat_orientation_l2,
                },
                "foot_air_time": {
                    "weight": 0.5,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "vel_cmd_manager": self.gait_command_manager,
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
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
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
            # history_len=4,
            cfg={
                "gait_cmd": {"fn": self.gait_command_manager.observation},
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
                    "scale": 0.05,
                    "noise": 0.1,
                },
                "dofs_force": {
                    "fn": lambda env: self.action_manager.get_dofs_force(
                        clip_to_max_force=True
                    ),
                    "scale": 0.1,
                    "noise": 0.01,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.raw_actions,
                },
            },
        )
        ObservationManager(
            self,
            # history_len=4,
            name="critic",
            cfg={
                "gait_cmd": {
                    "fn": self.gait_command_manager.privileged_observation,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                "air_time_target": {
                    "fn": self.air_time_observation,
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
        # dof_force = self.action_manager.get_dofs_force().abs()
        # control_force = self.robot.get_dofs_control_force(
        #     dofs_idx_local=self.action_manager.dofs_idx
        # ).abs()
        extras["episode"]["Metrics / curriculum_level"] = self._curriculum_level
        # extras["episode"]["Metrics / avg_actual_force"] = torch.mean(dof_force)
        # extras["episode"]["Metrics / avg_control_force"] = torch.mean(control_force)
        # extras["episode"]["Metrics / avg_air_time"] = torch.mean(
        #     self.foot_contact_manager.last_air_time
        # )

        if self.mode == "play":
            self.camera.render()

        # Finish up
        return obs, reward, terminated, truncated, extras

    def reset(self, envs_idx: list[int] | None = None):
        reset = super().reset(envs_idx)
        if envs_idx is not None:
            self.update_curriculum()
        return reset

    def air_time_observation(self, env: GenesisEnv) -> float:
        params = self.reward_manager.cfg["foot_air_time"].params
        obs = torch.zeros(env.num_envs, 1, device=gs.device)
        obs[:, 0] = (params["time_threshold"] + params["time_threshold_max"]) / 2.0
        return obs

    def inc_value(self, value: float, cfg: IncConfig):
        value += cfg["inc"]
        if cfg["limit"] is not None:
            if cfg["inc"] > 0:
                value = min(value, cfg["limit"])
            else:
                value = max(value, cfg["limit"])
        return value

    def inc_reward_param(self, reward_name: str, param_name: str, cfg: IncConfig):
        value = self.reward_manager.cfg[reward_name].params[param_name]
        self.reward_manager.cfg[reward_name].params[param_name] = self.inc_value(
            value, cfg
        )

    def inc_reward_weight(self, reward_name: str, cfg: IncConfig):
        value = self.reward_manager.cfg[reward_name].weight
        self.reward_manager.cfg[reward_name].weight = self.inc_value(value, cfg)

    def update_curriculum(self):
        """
        Check the curriculum
        """
        # Limit how often we check/update the curriculum
        if self.step_count < self._next_curriculum_check_step:
            return
        self._next_curriculum_check_step = (
            self.step_count + CURRICULUM_CHECK_EVERY_STEPS
        )

        cmd_linear_vel = self.reward_manager.last_episode_mean_reward(
            "cmd_linear_vel", before_weight=True
        )
        if cmd_linear_vel > 0.7:
            self._curriculum_level += 1
            self.gait_command_manager.increase_velocity()
            # self.gait_command_manager.jumping_probability = 0.1
            self.inc_reward_weight("similar_to_default", {"inc": 0.001, "limit": 0.01})
            self.inc_reward_weight("foot_sync", {"inc": 0.05, "limit": 0.5})

            self.inc_reward_weight("foot_air_time", {"inc": 0.05, "limit": 1.0})
            self.inc_reward_param(
                "foot_air_time", "time_threshold", {"inc": 0.05, "limit": 0.3}
            )
            self.inc_reward_param(
                "foot_air_time", "time_threshold_max", {"inc": 0.05, "limit": 1.0}
            )
