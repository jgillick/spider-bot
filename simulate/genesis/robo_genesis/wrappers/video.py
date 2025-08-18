import os
import math
import torch
from genesis.vis.camera import Camera
from typing import TypedDict, Tuple, Literal, Any, Sequence

from robo_genesis.wrappers.wrapper import Wrapper
from robo_genesis.genesis_env import GenesisEnv


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
}


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training.
    """

    cam: Camera = None
    follow_robot: VideoFollowRobotConfig = None
    out_dir: str
    current_step: int = 0
    every_n_steps: int
    video_length_steps: int
    is_recording: bool = False
    recording_steps_remaining: int = 0
    next_start_step: int = 0
    camera_config: VideoCameraConfig = DEFAULT_CAMERA
    filename: str = None

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

        self.out_dir = out_dir
        self.every_n_steps = every_n_steps
        self.video_length_steps = math.ceil(video_length_s / self.dt)
        self.camera_config = {**DEFAULT_CAMERA, **camera}
        self.filename = filename
        self.follow_robot = follow_robot

        os.makedirs(self.out_dir, exist_ok=True)

    def construct_scene(self):
        """Add a camera to the scene."""
        scene = super().construct_scene()
        self.cam = scene.add_camera(**self.camera_config)

    def build_scene(self):
        """Setup the camera to follow the robot."""
        super().build_scene()
        if self.follow_robot:
            self.cam.follow_entity(self.env.robot, **self.follow_robot)

    def start_recording(self):
        """Start recording a video."""
        self.is_recording = True
        self.recording_steps_remaining = self.video_length_steps
        self.cam.start_recording()
        self.cam.render()

    def finish_recording(self):
        """Stop recording and save the video."""
        if not self.is_recording:
            return

        # Save recording
        filename = self.filename or f"{self.next_start_step}.mp4"
        filepath = os.path.join(self.out_dir, filename)
        self.cam.stop_recording(filepath, fps=60)

        # Reset recording state
        self.is_recording = False
        self.recording_steps_remaining = 0
        self.next_start_step = self.current_step + self.every_n_steps

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Record a video image at each step."""
        self.current_step += 1

        # Currently recording
        if self.is_recording:
            self.cam.render()
            self.recording_steps_remaining -= 1
            if self.recording_steps_remaining <= 0:
                self.finish_recording()

        # Start new recording
        elif self.next_start_step <= self.current_step:
            self.start_recording()

        return super().step(actions)

    def close(self):
        """Finish recording on close"""
        if self.is_recording:
            self.finish_recording()
        super().close()
