import os
import torch
from typing import Literal
import genesis as gs

from genesis_forge import ManagedEnvironment, GenesisEnv, terrain
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
                camera_pos=(-2.25, -1.25, 1.25),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                refresh_rate=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
                enable_neutral_collision=False,
                batch_links_info=True,
                multiplier_collision_broad_phase=32,
                max_collision_pairs=40,
            ),
        )

        # Create terrain
        subterrains = []
        if terrain_type == "flat" or terrain_type == "mixed":
            subterrains += [terrain.Flat(name="flat")]
        if terrain_type == "rough" or terrain_type == "mixed":
            subterrains += [
                terrain.RandomUniform(
                    name="rough",
                    min_height=0.0,
                    max_height=0.08,
                    step=0.04,
                    downsampled_scale=0.25,
                    border_width=0.5,
                ),
                terrain.RandomGrid(
                    name="box",
                    grid_width=0.25,
                    grid_height=0.04,
                    platform_width=0.0,
                    horizontal_scale=0.0625,
                )
            ]
        self.terrain_layout = terrain.TerrainLayout(
            self.scene,
            pos=(-12, -12, 0),
            subterrain_size=(24, 24),
            vertical_scale=0.001,
            grid=[subterrains],
        )
        self.terrain = self.terrain_layout.entity

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
            pos=(-2.5, -1.5, 1.25),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot, smoothing=0.05)

        # Follow the robot in the viewer. (A GUI camera window would open an OpenCV/Qt
        # window, which deadlocks with the viewer thread and freezes the process.)
        if self.mode == "play" and not self.headless:
            self.scene.viewer.follow_entity(self.robot, smoothing=0.05)

        return self.scene

    def config(self):
        """
        Configure the environment managers.
        """
        if self.terrain_layout is not None:
            self.terrain_manager = TerrainManager(self, layout=self.terrain_layout)
        else:
            self.terrain_manager = TerrainManager(self, terrain=self.terrain)

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        obs, reward, terminated, truncated, extras = super().step(actions)

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
