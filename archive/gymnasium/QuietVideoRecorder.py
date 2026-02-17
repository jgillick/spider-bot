from typing import Callable

from gymnasium import logger

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
)
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder


class QuietVideoRecorder(VecVideoRecorder):
    """
    Wraps a VecVideoRecorder and silences the video progress bar.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    video_name: str
    video_path: str

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        super().__init__(
            venv,
            video_folder,
            record_video_trigger,
            video_length,
            name_prefix,
        )

    # def _start_video_recorder(self) -> None:
    #     super()._start_video_recorder()
    #     self.video_name = f"{self.name_prefix}.mp4"

    def _stop_recording(self) -> None:
        """Stop current recording and saves the video."""
        assert (
            self.recording
        ), "_stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:  # pragma: no cover
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.video_path, logger=None)

        self.recorded_frames = []
        self.recording = False
