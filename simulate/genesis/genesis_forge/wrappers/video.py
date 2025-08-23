import os
import math
import torch
from genesis.vis.camera import Camera
from typing import Tuple, Any

from genesis_forge.wrappers.wrapper import Wrapper
from genesis_forge.genesis_env import GenesisEnv


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training at a regular step intervals.
    To use this, you need to define a camera in the environment and assign it to a public attribute.
    When you wrap the environment, you pass the name of the attribute to the wrapper (see the example below).

    Args:
        env: GenesisEnv
        camera_attr: The attribute of the base environment that contains the camera to use for recording.
        video_length_s: Length of each video, in seconds.
        every_n_steps: Interval between each recording (in steps).
        out_dir: Directory to save the videos to.
        filename: The filename for the video.
                  If None, the video will automatically be named for the current step.
                  If defined, each video will overwrite the previous video with this name.

    Example::
        class MyEnv(GenesisEnv):
            camera: Camera

            def construct_scene(self) -> gs.Scene:
                scene = super().construct_scene()
                # Add robot...

                # Assign a camera to the `camera` attribute
                self.camera = scene.add_camera(pos=(-2.5, -1.5, 1.0))

                return scene

        def train():
            env = MyEnv()
            env = VideoWrapper(
                env,
                camera_attr="camera",
                out_dir="./videos"
            )
            env.build()
            ...training code...
    """

    def __init__(
        self,
        env: GenesisEnv,
        camera_attr: str = "camera",
        video_length_s: int = 8,
        every_n_steps: int = 500,
        out_dir: str = "./videos",
        filename: str = None,
    ):
        super().__init__(env)
        self._cam = None
        self._camera_attr = camera_attr
        self._out_dir = out_dir
        self._filename = filename
        self._every_n_steps = every_n_steps
        self._video_length_steps = math.ceil(video_length_s / self.dt)
        self._current_step: int = 0
        self._is_recording: bool = False
        self._recording_steps_remaining: int = 0
        self._next_start_step: int = 0

        os.makedirs(self._out_dir, exist_ok=True)

    def build(self) -> None:
        """Load the camera from the environment."""
        super().build()
        self._cam = self.unwrapped.__getattribute__(self._camera_attr)
        assert (
            self._cam is not None
        ), f"Camera not found at attribute: {self.unwrapped.__class__.__name__}.{self._camera_attr}"

    def start_recording(self):
        """Start recording a video."""
        self._is_recording = True
        self._recording_steps_remaining = self._video_length_steps
        self._cam.start_recording()
        self._cam.render()

    def finish_recording(self):
        """Stop recording and save the video."""
        if not self._is_recording and self._cam is not None:
            return

        # Save recording
        filename = self._filename or f"{self._next_start_step}.mp4"
        filepath = os.path.join(self._out_dir, filename)
        self._cam.stop_recording(filepath, fps=60)

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
            self._cam.render()
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
