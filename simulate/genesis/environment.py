"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import math
import torch
import numpy as np
from PIL import Image
from typing import Any
from gymnasium import spaces
import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat

from genesis_forge import GenesisEnv, ManagedEnvironment, EnvMode
from genesis_forge.managers import (
    CommandManager,
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
    PositionalActionManager,
    ContactManager,
    TerrainManager,
)
from genesis_forge.utils import (
    entity_projected_gravity,
    entity_ang_vel,
    entity_lin_vel,
    links_idx_by_name_pattern,
)
from genesis_forge.mdp import rewards, terminations


INITIAL_BODY_POSITION = [0.0, 0.0, 0.14]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


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
        terrain: str = "flat_terrain",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
            headless=headless,
        )
        self.camera = None
        self._curriculum_phase = 1
        self._training_terrain = terrain

        # Cache position buffers
        self.base_init_pos = torch.tensor(INITIAL_BODY_POSITION, device=gs.device)
        self.base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device).reshape(
            1, -1
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.rand_quat_rotation = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        """
        Configuration
        """
        # Terrain
        self.terrain_manager = TerrainManager(self, terrain_attr="terrain")

        # Define the DOF actuators
        self.action_manager = PositionalActionManager(
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
            # stiffness=0.1,
        )

        # Command manager: instruct the robot to move in a certain direction
        self.height_command = CommandManager(
            self,
            range={
                "height": [0.12, 0.15],
            },
        )
        self.velocity_command = VelocityCommandManager(
            self,
            # Starting ranges should be small, while robot is learning to stand
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-1.0, 1.0],
            },
            standing_probability=0.02,
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        # Contact managers
        self.foot_contact_manager = ContactManager(
            self,
            link_names=["Leg[1-8]_Tibia_Foot"],
            with_entity_attr="terrain",
            track_air_time=True,
        )
        self.self_contact = ContactManager(
            self,
            link_names=[
                "Leg[1-8]_Femur",
                "Leg[1-8]_Tibia_Leg",
            ],
            with_entity_attr="robot",
            with_links_names=[
                "Leg[1-8]_Femur",
                "Leg[1-8]_Tibia_Leg",
            ],
        )

        # Rewards
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "Linear Z velocity": {
                    "weight": -10.0,
                    "fn": rewards.lin_vel_z,
                },
                "Base height": {
                    "weight": -1000.0,
                    "fn": rewards.base_height,
                    "params": {
                        # "target_height": 0.135,
                        "height_command": self.height_command,
                        "terrain_manager": self.terrain_manager,
                    },
                },
                "Similar to default": {
                    "weight": -0.5,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "dof_action_manager": self.action_manager,
                    },
                },
                "Action rate": {
                    "weight": -0.05,
                    "fn": rewards.action_rate,
                },
                "Cmd linear velocity": {
                    "weight": 20.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Cmd angular velocity": {
                    "weight": 10.0,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Flat orientation": {
                    "weight": -50.0,
                    "fn": rewards.flat_orientation_l2,
                },
                "Self contact": {
                    "weight": -10.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.self_contact,
                    },
                },
                # "Self contact": {
                #     "weight": -0.012,
                #     "fn": rewards.contact_force,
                #     "params": {
                #         "contact_manager": self.self_contact,
                #         "threshold": 0.1,
                #     },
                # },
                "Foot air time": {
                    "weight": 1.25,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "vel_cmd_manager": self.velocity_command,
                        "threshold": 1.0,
                    },
                },
                "Leg angle": {
                    "weight": -2.0,
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

        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "Timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "Bad angle": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 0.7,
                    },
                },
                # "Self contact force": {
                #     "fn": terminations.contact_force,
                #     "params": {
                #         "contact_manager": self.self_contact,
                #         "threshold": 30.0,
                #     },
                # },
            },
        )

    """
    Properties
    """

    @property
    def action_space(self):
        return self.action_manager.action_space

    """
    Operations
    """

    def construct_scene(self) -> gs.Scene:
        """
        Construct the environment scene.
        """
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 1.0),
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
        self.terrain = self.scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=tiled_image,
                )
            ),
            morph=gs.morphs.Terrain(
                n_subterrains=(1, 3),
                subterrain_size=(12, 12),
                vertical_scale=0.002,
                subterrain_types=[
                    [
                        "flat_terrain",
                        # "fractal_terrain", 
                        "discrete_obstacles_terrain",
                        "pyramid_stairs_terrain",
                    ],
                    # ["flat_terrain", "pyramid_stairs_terrain"],
                    # ["discrete_obstacles_terrain", "fractal_terrain"],   # "random_uniform_terrain"
                ],
            ),
        )
        # self.terrain = self.scene.add_entity(gs.morphs.Plane())

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Add camera
        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            res=(1280, 960),
            fov=40,
            env_idx=0,
            debug=True,
        )

        return self.scene

    def build(self) -> None:
        """
        Builds the scene after all entities have been added (via construct_scene).
        This operation is required before running the simulation.
        """
        super().build()
        
        # Set observation space from first observation
        if self.observation_space is None:
            obs = self.observations()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs.shape[1],),
                dtype=np.float32,
            )

        # Track robot with camera
        self.camera.follow_entity(self.robot, fixed_axis=(None, None, 1.0))
        self.camera.set_pose(lookat=self.robot.get_pos())

        # Fetch foot links
        self._foot_links_idx = links_idx_by_name_pattern(
            self.robot, "Leg[1-8]_Tibia_Foot"
        )
        self._foot_link_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self._foot_link_gravity = self._foot_link_gravity.unsqueeze(0).expand(
            self.num_envs, len(self._foot_links_idx), 3
        )

    def observations(self) -> torch.Tensor:
        """Generate a list of observations for each environment."""

        # If this is being called before the first step, actions should all be zero
        actions = self.actions
        if actions is None:
            actions = torch.zeros(
                (self.num_envs, self.action_manager.num_actions), device=gs.device
            )

        # Create observations
        return torch.cat(
            [
                self.height_command.command,  # 1
                self.velocity_command.command,  # 3
                entity_ang_vel(self.robot),  # 3
                entity_lin_vel(self.robot),  # 3
                entity_projected_gravity(self.robot),  # 3
                self.action_manager.get_dofs_position(noise=0.01),  # 24
                self.action_manager.get_dofs_velocity(noise=0.1),  # 24
                actions,  # 24
                self.action_manager.get_dofs_force(
                    noise=0.01, clip_to_max_force=True
                ),  # 24
            ],
            dim=-1,
        )

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        _, reward, terminated, truncated, extras = super().step(actions)
        obs = self.observations()

        #  Keep the camera looking at the robot
        self.camera.set_pose(lookat=self.robot.get_pos())

        # Update curriculum
        self._update_curriculum()

        # Log metrics
        extras["episode"]["Metrics / Self Contact"] = torch.mean(
            rewards.has_contact(self, self.self_contact)
        )
        extras["episode"]["Metrics / Foot Contact"] = torch.mean(
            rewards.has_contact(self, self.foot_contact_manager)
        )
        extras["episode"]["Metrics / Curriculum level"] = torch.tensor(
            self._curriculum_phase, device="cpu"
        )

        # Finish up
        return obs, reward, terminated, truncated, extras

    def reset(
        self,
        envs_idx: list[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or more environments.
        """
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # Reset robot
        if envs_idx.numel() > 0:
            # Randomize positions on the terrain
            pos = self.terrain_manager.generate_random_env_pos(
                envs_idx=envs_idx,
                subterrain=self._training_terrain,
                height_offset=0.15,
            )

            # Randomize quat rotation
            self.rand_quat_rotation[envs_idx] = (
                torch.rand(len(envs_idx), device=gs.device) * 2 * math.pi
            )
            self.base_quat[envs_idx] = self.base_init_quat
            self.base_quat[envs_idx, 0] = torch.cos(self.rand_quat_rotation[envs_idx])
            self.base_quat[envs_idx, 3] = torch.sin(self.rand_quat_rotation[envs_idx])

            # Send to robot
            self.robot.zero_all_dofs_velocity(envs_idx)
            self.robot.set_pos(pos, envs_idx=envs_idx)
            self.robot.set_quat(self.base_quat[envs_idx], envs_idx=envs_idx)

        obs = self.observations()
        return obs, {}

    def render(self):
        pass

    def close(self):
        """Close the environment."""
        self.scene.reset()

    """
    Implementation
    """

    def _update_curriculum(self):
        """
        Update the training curriculum based on the current step or environment performance.
        """
        # Move to more uneven terrain
        if self.step_count == 50_000:
            self._training_terrain = "discrete_obstacles_terrain"
            self.reward_manager.cfg["Base height"]["weight"] = -1.0
            self.reward_manager.cfg["Self contact"]["weight"] = -1.0
            self.set_max_episode_length(round(self.max_episode_length_sec * 1.5))
        
        # Open all terrains
        elif self.step_count == 60_000:
            # Unsetting the terrain will open all subterrains
            self._training_terrain = None
            
        # Pick up the pace
        elif self.step_count == 80_000:
            self.velocity_command.range["lin_vel_x"] = [-1.5, 1.5]
            self.velocity_command.range["lin_vel_y"] = [-1.5, 1.5]
            self.velocity_command.range["ang_vel_z"] = [-1.5, 1.5]
            # self.reward_manager.cfg["Similar to default"]["weight"] = 0.5
            # self.reward_manager.cfg["Bad touch"]["weight"] = -10.0
            # self.reward_manager.cfg["Foot contact (some)"]["weight"] = 5
            # self.reward_manager.cfg["Foot contact (all)"]["weight"] = 0.1

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
