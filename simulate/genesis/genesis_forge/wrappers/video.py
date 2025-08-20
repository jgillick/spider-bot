import os
import math
import torch
from genesis.vis.camera import Camera
from typing import TypedDict, Tuple, Literal, Any, Sequence

from genesis_forge.wrappers.wrapper import Wrapper
from genesis_forge.genesis_env import GenesisEnv


class VideoCameraConfig(TypedDict):
    """
    The camera configuration for the video that will be passed
    directly to scene.add_camera.
    @see https://genesis-world.readthedocs.io/en/latest/api_reference/scene/scene.html#genesis.engine.scene.Scene.add_camera
    """

    model: Literal["pinhole", "thinlens"]
    res: Tuple[int, int]
    pos: Tuple[float, float, float]
    lookat: Tuple[float, float, float]
    fov: int
    up: Tuple[float, float, float]
    aperture: float
    focus_dist: float
    spp: int
    denoise: bool
    env_idx: int
    debug: bool


class VideoFollowRobotConfig(TypedDict):
    """
    The "follow_entity" configuration for the camera to follow the robot
    """

    fixed_axis: Tuple[float, float, float]
    smoothing: float
    fix_orientation: bool


DEFAULT_CAMERA: VideoCameraConfig = {
    "model": "pinhole",
    "res": (1280, 960),
    "pos": (0.5, 2.5, 3.5),
    "lookat": (0.5, 0.5, 0.5),
    "fov": 40,
    "up": (0.0, 0.0, 1.0),
    "aperture": 2.0,
    "focus_dist": None,
    "spp": 256,
    "denoise": True,
    "env_idx": 0,
    "debug": True,
}


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training.
    """

    def __init__(
        self,
        env: GenesisEnv,
        every_n_steps: int = 500,
        video_length_s: int = 5,
        out_dir: str = "videos",
        camera: VideoCameraConfig = DEFAULT_CAMERA,
        follow_robot: VideoFollowRobotConfig = None,
        filename: str = None,
    ):
        super().__init__(env)

        self._out_dir = out_dir
        self._filename = filename
        self._every_n_steps = every_n_steps
        self._video_length_steps = math.ceil(video_length_s / self.dt)
        self._camera_config = {**DEFAULT_CAMERA, **camera}
        self._follow_robot = follow_robot
        self._current_step: int = 0
        self._is_recording: bool = False
        self._recording_steps_remaining: int = 0
        self._next_start_step: int = 0

        os.makedirs(self._out_dir, exist_ok=True)

    def construct_scene(self):
        """Add a camera to the scene."""
        scene = super().construct_scene()
        self.cam = scene.add_camera(**self._camera_config)

    def build(self):
        """Setup the camera to follow the robot."""
        super().build()
        if self._follow_robot:
            self.cam.follow_entity(self.env.robot, **self._follow_robot)

    def start_recording(self):
        """Start recording a video."""
        self._is_recording = True
        self._recording_steps_remaining = self._video_length_steps
        self.cam.start_recording()
        self.cam.render()

    def finish_recording(self):
        """Stop recording and save the video."""
        if not self._is_recording:
            return

        # Save recording
        filename = self._filename or f"{self._next_start_step}.mp4"
        filepath = os.path.join(self._out_dir, filename)
        self.cam.stop_recording(filepath, fps=60)

        # Reset recording state
        self._is_recording = False
        self._recording_steps_remaining = 0
        self._next_start_step = self._current_step + self._every_n_steps

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Record a video image at each step."""
        self._current_step += 1

        # Currently recording
        if self._is_recording:
            self.cam.render()
            self._recording_steps_remaining -= 1
            if self._recording_steps_remaining <= 0:
                self.finish_recording()

        # Start new recording
        elif self._next_start_step <= self._current_step:
            self.start_recording()

        return super().step(actions)

    def close(self):
        """Finish recording on close"""
        if self._is_recording:
            self.finish_recording()
        super().close()
