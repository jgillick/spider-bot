"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import re
import os
import torch
import numpy as np
from PIL import Image
from typing import Literal
import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat

from genesis_forge import GenesisEnv, ManagedEnvironment
from genesis_forge.managers import (
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
    PositionWithinLimitsActionManager,
    ContactManager,
    TerrainManager,
    EntityManager,
    ObservationManager,
)
from genesis_forge.mdp import reset, rewards, terminations


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


Terrain = Literal["flat", "rough", "mixed"]
EnvMode = Literal["train", "eval", "play"]


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
            max_episode_random_scaling=0.5,
        )
        self._curriculum_phase = 1
        self.headless = headless
        self.mode = mode
        self._next_curriculum_update = self.max_episode_length_steps
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
        )
        self.camera.follow_entity(self.robot, smoothing=0.05)

        return self.scene

    def config(self):
        """
        Configure the environment managers.
        """

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
            pd_kv=0.5,
            max_force=8.0,
            frictionloss=0.1,
            noise_scale=0.02,
        )

        ##
        # Command managers

        # Command manager: instruct the robot to move in a certain direction
        self.velocity_command = VelocityCommandManager(
            self,
            # Starting ranges should be small, while robot is learning to stand
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-1.0, 1.0],
            },
            standing_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
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
                # "*._Motor",
            ],
            with_entity_attr="robot",
            with_links_names=[
                "Leg[1-8]_Femur",
                "Leg[1-8]_Tibia_Leg",
                # "*._Motor",
            ],
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "size": 0.05,
            },
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(
            self,
            cfg={
                "Linear Z velocity": {
                    "weight": -0.1,
                    "fn": rewards.lin_vel_z_l2,
                },
                "Base height": {
                    "weight": -50.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.14,
                        "terrain_manager": self.terrain_manager,
                    },
                },
                "Similar to default": {
                    "weight": -0.05, 
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "action_manager": self.action_manager,
                    },
                },
                "Action rate": {
                    "weight": -5e-04,
                    "fn": rewards.action_rate_l2,
                },
                "Cmd linear velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Cmd angular velocity": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Flat orientation": {
                    "weight": -0.5,
                    "fn": rewards.flat_orientation_l2,
                },
                "Self contact": {
                    "weight": -15.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.self_contact,
                    },
                },
                # "Self contact": {
                #     "weight": -0.1, # -0.2
                #     "fn": rewards.contact_force,
                #     "params": {
                #         "contact_manager": self.self_contact,
                #         "threshold": 0.2,
                #     },
                # },
                "Foot air time": {
                    "weight": 0.2,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "vel_cmd_manager": self.velocity_command,
                        "time_threshold": 1.0,
                    },
                },
                "Leg angle": {
                    "weight": -0.02,
                    "fn": self._penalize_leg_angle,
                },
                "Foot contact (some)": {
                    "weight": 0.0,  # 5.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "min_contacts": 4,
                    },
                },
                "Foot contact (all)": {
                    "weight": 0.0,  # 0.1,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "min_contacts": 8,
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
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
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

    def build(self) -> None:
        """
        Builds the scene after all entities have been added (via construct_scene).
        This operation is required before running the simulation.
        """
        super().build()

        # Fetch foot links
        self._foot_links_idx = []
        for link in self.robot.links:
            if re.match("^Leg[1-8]_Tibia_Foot$", link.name):
                self._foot_links_idx.append(link.idx_local)
        self._foot_link_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self._foot_link_gravity = self._foot_link_gravity.unsqueeze(0).expand(
            self.num_envs, len(self._foot_links_idx), 3
        )

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        obs, reward, terminated, truncated, extras = super().step(actions)
        # self._update_curriculum(extras)

        # Log metrics
        # extras["episode"]["Metrics / Self Contact"] = torch.mean(
        #     rewards.has_contact(self, self.self_contact)
        # )
        # extras["episode"]["Metrics / Foot Contact"] = torch.mean(
        #     rewards.has_contact(self, self.foot_contact_manager)
        # )
        # extras["episode"]["Metrics / Curriculum level"] = torch.tensor(
        #     self._curriculum_phase, device="cpu"
        # )

        #  Keep the camera looking at the robot
        self.camera.set_pose(lookat=self.robot.get_pos()[0])
        # If we're playing a pre-trained agent, render the camera
        if self.mode == "play":
            self.camera.render()

        # Finish up
        return obs, reward, terminated, truncated, extras


    """
    Implementation
    """

    def _update_curriculum(self, extras: dict):
        """
        If the robot is able to walk, incrementally increase the velocity range.
        """
        # Don't update on every reset
        if self.step_count > self._next_curriculum_update:
            self._next_curriculum_update = self.step_count + self.max_episode_length_steps

            # If the reward is 80%+ of the max, increase the velocity range
            reward_name = "Cmd linear velocity"
            linear_velocity_reward = self.reward_manager.last_episode_mean_reward(reward_name)
            if linear_velocity_reward >= 0.8 * self.reward_manager.cfg[reward_name]["weight"]:
                print(f"Increasing velocity range {reward_name}: {linear_velocity_reward} >= {0.8 * self.reward_manager.cfg[reward_name]['weight']}")
                if self.velocity_command.range["lin_vel_x"][1] < 1.5:
                    self.velocity_command.range["lin_vel_x"][0] -= 0.1
                    self.velocity_command.range["lin_vel_x"][1] += 0.1
                if self.velocity_command.range["lin_vel_y"][1] < 1.0:
                    self.velocity_command.range["lin_vel_y"][0] -= 0.1
                    self.velocity_command.range["lin_vel_y"][1] += 0.1
                    self.velocity_command.range["ang_vel_z"][0] -= 0.1
                    self.velocity_command.range["ang_vel_z"][1] += 0.1

            extras["episode"]["Metrics / Linear velocity target"] = self.velocity_command.range["lin_vel_x"][1]
        

    def _penalize_leg_angle(self, _env: GenesisEnv):
        """
        Penalize the tibia bending in too far and under the robot.
        The penalty is the sum of how far the projected gravity of each leg is below zero.
        """
        target_angle = 0.0  # penalize anything less than this

        quats = self.robot.get_links_quat(links_idx_local=self._foot_links_idx)

        # Transform link frames to world gravity frames
        inv_quats = inv_quat(quats)
        gravity_in_links = transform_by_quat(self._foot_link_gravity, inv_quats)
        uprightness = -gravity_in_links[..., 2]

        # Add up all the uprightness values less than zero
        return torch.abs(torch.sum(uprightness * (uprightness < target_angle), dim=1))
