import torch
import genesis as gs
from typing import Sequence, TypedDict, Callable, Any

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class RewardConfig(TypedDict):
    """Defines a reward item."""

    fn: Callable[[GenesisEnv, ...], torch.Tensor]
    """Function that will be called to calculate a reward for the environments."""

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    weight: float
    """The weight of the reward item."""


class RewardManager(BaseManager):
    """
    Handles calculating and logging the rewards for the environment.

    This works with a dictionary configuration of reward handlers. For each dictionary item,
    a function will be called to calculate a reward value for the environment.

    Args:
        env: The environment to manage the rewards for.
        reward_cfg: A dictionary of reward conditions.
        logging_enabled: Whether to log the rewards to tensorboard.

    Example:
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def configuration_managers(self):
                self.reward_manager = RewardManager(
                    self,
                    term_cfg={
                        "Default pose": {
                            "fn": mdp.rewards.dof_similar_to_default,
                            "weight": -0.1,
                        },
                        "Base height": {
                            "fn": mdp.rewards.base_height,
                            "params": { "target_height": 0.135 },
                            "weight": -100.0,
                        },
                    },
                )

            def step(self, actions: torch.Tensor):
                super().step(actions)
                # ...handle actions...

                # Termination manager should be called before reward manager

                self.reward_manager.step()

                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: Sequence[int] = None):
                super().reset(envs_idx)
                # ...do reset logic here...x

                self.reward_manager.reset(envs_idx)
                return obs, info

    """

    def __init__(
        self,
        env: GenesisEnv,
        cfg: dict[str, RewardConfig],
        logging_enabled: bool = True,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.logging_enabled = logging_enabled

        self._reward_buf = torch.zeros(
            (env.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self._episode_length = torch.zeros(
            (self.env.num_envs,), device=gs.device, dtype=torch.int32
        )
        self._episode_data: dict[str, torch.Tensor] = dict()
        for name in self.cfg.keys():
            self._episode_data[name] = torch.zeros(
                (env.num_envs,), device=gs.device, dtype=gs.tc_float
            )

    def step(self) -> torch.Tensor:
        """
        Calculate the rewards for this step

        Returns:
            The rewards for the environments. Shape is (num_envs,).
        """
        self._reward_buf[:] = 0.0
        self._episode_length += 1
        for name, cfg in self.cfg.items():
            fn = cfg["fn"]
            weight = cfg.get("weight", 0.0)
            params = cfg.get("params", dict())

            # Don't calculate reward if the weight is zero
            if weight == 0:
                continue
            weight = weight * self.env.dt

            value = fn(self.env, **params) * weight
            self._reward_buf += value
            self._episode_data[name] += value

        return self._reward_buf

    def reset(self, envs_idx: Sequence[int] = None):
        """Log the reward mean values at the end of the episode"""
        if not self.logging_enabled:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        episode_lengths = self._episode_length[envs_idx]
        valid_episodes = episode_lengths > 0
        for name, value in self._episode_data.items():
            # Log episodes with at least one step (otherwise it could cause a divide by zero error)
            if torch.any(valid_episodes):
                # Calculate average for each episode based on its actual length
                episode_avg = torch.zeros_like(value[envs_idx])
                episode_avg[valid_episodes] = (
                    value[envs_idx][valid_episodes] / episode_lengths[valid_episodes]
                )

                # Take the mean across valid episodes only
                episode_mean = torch.mean(episode_avg[valid_episodes]).cpu().item()
                self.env.track_data(f"Rewards / {name}", episode_mean)

            # Reset episodic sum
            self._episode_data[name][envs_idx] = 0.0

        self._episode_length[envs_idx] = 0
