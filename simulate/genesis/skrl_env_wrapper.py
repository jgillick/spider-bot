from typing import Any, Tuple, Callable

import torch
from skrl.envs.wrappers.torch.base import Wrapper


class SkrlEnvWapper(Wrapper):

    def set_data_tracker(self, track_data_fn: Callable[[str, float], None]):
        """Set the function which logs data to tensorboard."""
        self._env.set_data_tracker(track_data_fn)

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        return self._env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        obs, rewards, terminations, timeouts, extras = self._env.step(actions)

        # Expand rewards, terminations and timeouts to the shape (num_envs, 1)
        rewards = rewards.unsqueeze(1)
        terminations = terminations.unsqueeze(1)
        timeouts = timeouts.unsqueeze(1)

        return obs, rewards, terminations, timeouts, extras

    def state(self) -> torch.Tensor:
        """Get the environment state

        :return: State
        :rtype: torch.Tensor
        """
        return self._env.state()

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :return: Any value from the wrapped environment
        :rtype: any
        """
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        return self._env.close()
