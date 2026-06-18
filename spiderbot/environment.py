import os
import re
import torch
import numpy as np
from PIL import Image
from typing import Literal
import genesis as gs

from genesis_forge import ManagedEnvironment, GenesisEnv
from genesis_forge.managers import (
    TerrainManager,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../model/v2/SpiderBot.xml"))

Terrain = Literal["flat", "rough", "mixed"]
EnvMode = Literal["train", "eval", "play"]


class BaseSpiderRobotEnv(ManagedEnvironment):
    """
    SpiderBot environment for Genesis
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_sec: int | None = 6,
        headless: bool = True,
        mode: EnvMode = "train",
        terrain: Terrain = "flat",
        height_sensor: bool = False,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_sec,
            # max_episode_random_scaling=0.1,
        )
        self.headless = headless
        self.mode = mode
        self.use_height_sensor = height_sensor
        self.terrain_type = terrain

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
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
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
                enable_neutral_collision=True,
                multiplier_collision_broad_phase=32,
                max_collision_pairs=40,
            ),
        )

        # Create terrain
        checker_path = os.path.join(THIS_DIR, "assets/checker.png")
        checker_image = np.array(Image.open(checker_path))
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
                pos=[0.0, 0.0, 0.118],
                quat=[1.0, 0.0, 0.0, 0.0],
            ),
        )

        # IMU Sensor
        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                pos_offset=(0.24, 0.0, 0.0),
                euler_offset=(0.0, 0.0, 0.0),
                acc_noise=(0.01, 0.01, 0.01),
                gyro_noise=(0.01, 0.01, 0.01),
                acc_random_walk=(0.001, 0.001, 0.001),
                gyro_random_walk=(0.001, 0.001, 0.001),
                delay=self.dt,
                jitter=self.dt,
            )
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
        self.terrain_manager = TerrainManager(self, terrain_attr="terrain")

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        obs, reward, terminated, truncated, extras = super().step(actions)

        if self.mode == "play":
            self.camera.render()

        # Finish up
        return obs, reward, terminated, truncated, extras

    def height_sensor_observation(self, env: GenesisEnv) -> torch.Tensor:
        # Get height above terrain from simulator
        if self.height_sensor is None:
            base_pos = self.robot.get_pos()
            height_offset = self.terrain_manager.get_terrain_height(
                base_pos[:, 0], base_pos[:, 1]
            )
            height = base_pos[:, 2] - height_offset
            return height.unsqueeze(-1)  # (n_envs, 1)

        # Get the height grid from lidar sensor
        heights = self.height_sensor.read().distances
        return heights.flatten(start_dim=-2)  # (n_envs, 5, 5) -> (n_envs, 25)

    def imu_observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        Makes an IMU reading and returns the concatenated linear acceleration and angular velocity readings.

        Returns:
            torch.Tensor: Shape (n_envs, 6): [lin_acc_xyz, ang_vel_xyz] per env.
                        Shape: (num_envs, 6)
        """
        value = self.imu.read()
        return torch.cat([value.lin_acc, value.ang_vel], dim=-1)
