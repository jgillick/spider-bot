import torch
import genesis as gs
from typing import Any, Tuple, Callable, Union, Sequence

from robo_genesis.wrappers.wrapper import Wrapper
from robo_genesis.genesis_env import GenesisEnv

LogFn = Union[Callable[[str, float], None], None]


class DataLoggerWrapper(Wrapper):
    """
    Wrapper that automatically log data from the info dict.
    IMPORTANT: This needs to be the outermost Genesis wrapper, otherwise some data will not be logged.

    Data can either be logged at each step, or at the end of each episode (at reset).

    Example:
    ```python
        def step(self, actions):
            # ... Do some step stuff

            # Initialize the logs dict
            info["logs"] = {}

            # Step logs
            info["logs"]["step"] = {}
            info["logs"]["step"]["Step / My Chart"] = Tensor.zeros((num_envs,))
            info["logs"]["step"]["Step / My Chart"][0] = 1.0

            # Episode logs
            info["logs"]["episode"] = {}
            info["logs"]["episode"]["Episode / My Chart"] = Tensor.zeros((num_envs,))
            info["logs"]["episode"]["Episode / My Chart"][0] = 1.0

            # Return
            return obs, rewards, terminations, timeouts, info
    ```
    """

    key: str
    log_fn: LogFn = None
    episode_data: dict[str, torch.Tensor] = None

    def __init__(self, env: GenesisEnv, logger_fn: LogFn = None, log_key: str = "logs"):
        """Initialize the logging wrapper with the function to use for data logging."""
        super().__init__(env)
        self.key = log_key
        self.log_fn = logger_fn
        self.episode_length = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.episode_data = dict()

    def set_data_tracker(self, track_data_fn: Callable[[str, float], None]):
        """Set the function which logs data to tensorboard."""
        self.log_fn = track_data_fn

    def track_data(self, name: str, value: float):
        """Log a single value to the logger function."""
        if not self.log_fn:
            print(f"Warning: No logger function set for logging data.")
            return
        self.log_fn(name, value)

    def log_episode_data(self, env_ids: Sequence[int]):
        """Log episode values."""
        if self.episode_data is None:
            return

        episode_lengths = self.episode_length[env_ids]
        valid_episodes = episode_lengths > 0
        for name, value in self.episode_data.items():
            # Log episodes with at least one step
            if torch.any(valid_episodes):
                if not isinstance(value, torch.Tensor):
                    print(
                        f"Warning: Episode log value for '{name}' is not a torch.Tensor"
                    )
                    continue
                if value.numel() < self.num_envs:
                    print(
                        f"Warning: Episode log value for '{name}' does not have the shape ({self.num_envs},)"
                    )
                    continue

                # Calculate average for each episode based on its actual length
                episode_avg = torch.zeros_like(value[env_ids])
                episode_avg[valid_episodes] = (
                    value[env_ids][valid_episodes] / episode_lengths[valid_episodes]
                )

                # Take the mean across valid episodes only
                episode_mean = torch.mean(episode_avg[valid_episodes])
                self.track_data(name, episode_mean)

            # Reset episodic sum
            self.episode_data[name][env_ids] = 0.0

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment and log data at the end of it."""
        self.episode_length += 1
        obs, rewards, terminations, timeouts, info = super().step(actions)

        if self.key in info:
            # Step logs
            if "step" in info[self.key]:
                for name, value in info[self.key]["step"].items():
                    if isinstance(value, torch.Tensor):
                        value = value.mean().item()
                    self.track_data(name, value)

            # Update episode logs
            if "episode" in info[self.key]:
                for name, value in info[self.key]["episode"].items():
                    self.episode_data[name] = value

        # Log episode data for environments that reset at this step
        resets = timeouts | terminations
        reset_idx = resets.nonzero(as_tuple=False).reshape((-1,))
        if reset_idx.numel() > 0:
            self.log_episode_data(reset_idx)

        return obs, rewards, terminations, timeouts, info

    def reset(
        self, env_ids: Sequence[int] = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        (obs, info) = super().reset(env_ids)

        # If env_ids is None, we're resetting all environments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=gs.device)

        # Episode logs
        if self.key in info and "episode" in info[self.key]:
            for name, value in info[self.key]["episode"].items():
                self.episode_data[name] = value
        self.log_episode_data(env_ids)

        return obs, info
