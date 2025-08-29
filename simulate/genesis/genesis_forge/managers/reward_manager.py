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

    def step(self) -> torch.Tensor:
        """
        Calculate the rewards for this step

        Returns:
            The rewards for the environments. Shape is (num_envs,).
        """
        self._reward_buf[:] = 0.0
        if not self.enabled:
            return self._reward_buf

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
            if self.logging_enabled:
                self.env.track_data(f"Rewards / {name}", value.mean().cpu().item())

        return self._reward_buf
