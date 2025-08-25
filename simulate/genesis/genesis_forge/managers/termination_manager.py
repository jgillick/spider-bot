import torch
from typing import Sequence, TypedDict, Callable, Any

import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class TerminationConfig(TypedDict):
    """Defines a termination condition."""

    fn: Callable[[GenesisEnv, ...], torch.Tensor]
    """Function that will be called to calculate a termination signal for the environment."""

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    time_out: bool
    """Set to True if a positive result is a time out and not a termination."""


class TerminationManager(BaseManager):
    """
    Handles calculating and logging the "dones" (termination or truncation) for the environments.

    This works with a dictionary configuration of termination conditions. For each dictionary item,
    a function will be called to calculate a termination signal for the environment.

    Args:
        env: The environment to manage the termination for.
        term_cfg: A dictionary of termination conditions.
        logging_enabled: Whether to log the termination signals to tensorboard.

    Example:
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def configuration_managers(self):
                self.termination_manager = TerminationManager(
                    self,
                    term_cfg={
                        "Min Height": {
                            "fn": mdp.terminations.min_height,
                            "params": {"min_height": 0.5},
                        },
                        "Rolled over": {
                            "fn": mdp.terminations.max_angle,
                            "params": { "quat_threshold": 0.35 },
                        },
                    },
                )

            def step(self, actions: torch.Tensor):
                super().step(actions)
                # ...handle actions...

                terminated, truncated, reset_env_idx = self.termination_manager.step()

                # reward manager should be called after termination manager, especially if you have alive/termination rewards

                self.reset(reset_env_idx)
                return obs, rewards, terminated, truncated, info

            def reset(self, envs_idx: Sequence[int] = None):
                super().reset(envs_idx)
                # ...do reset logic here...x

                self.termination_manager.reset(envs_idx)
                return obs, info

    """

    def __init__(
        self,
        env: GenesisEnv,
        term_cfg: dict[str, TerminationConfig],
        logging_enabled: bool = True,
    ):
        super().__init__(env)
        self.term_cfg = term_cfg
        self.logging_enabled = logging_enabled
        self._terminated_buf = torch.zeros(
            env.num_envs, device=gs.device, dtype=torch.bool
        )
        self._truncated_buf = torch.zeros_like(self._terminated_buf)
        self._term_data: dict[str, torch.Tensor] = dict()

    @property
    def dones(self) -> torch.Tensor:
        """The termination signals for the environments. Shape is (num_envs,)."""
        return self._terminated_buf | self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The termination signals for the environments. Shape is (num_envs,)."""
        return self._terminated_buf

    @property
    def truncated(self) -> torch.Tensor:
        """The truncation signals for the environments. Shape is (num_envs,)."""
        return self._truncated_buf

    def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the termination/truncation signals for this step

        Returns:
            terminated - The termination signals for the environments. Shape is (num_envs,).
            truncated - The truncation signals for the environments. Shape is (num_envs,).
            reset_env_idx - The indices of the environments that need to be reset.
        """
        self._term_data = dict()
        self._terminated_buf[:] = False
        self._truncated_buf[:] = False

        for name, cfg in self.term_cfg.items():
            fn = cfg["fn"]
            params = cfg.get("params", dict())
            trunc = cfg.get("time_out", False)

            # Get value and ensure it's boolean
            value = fn(self.env, **params)
            if value.dtype != torch.bool:
                print(
                    f"Warning: Termination function '{name}' returned {value.dtype} tensor, converting to bool"
                )
                value = value.bool()
            self._term_data[name] = value

            # Add to the correct buffer
            if trunc:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value

        # Return the environments that need to be reset
        dones = self._terminated_buf | self._truncated_buf
        reset_env_idx = dones.nonzero(as_tuple=False).reshape((-1,))
        return self._terminated_buf, self._truncated_buf, reset_env_idx

    def reset(self, env_ids: Sequence[int] = None):
        """Track terminated/truncated environments."""
        super().reset(env_ids)
        if not self.logging_enabled:
            return

        for name, value in self._term_data.items():
            self.env.track_data(f"Dones / {name}", value.float().mean().cpu().item())
