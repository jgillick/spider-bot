from typing import Any, TypeVar, Sequence, Callable

import torch
import genesis as gs
from gymnasium import spaces
from robo_genesis.genesis_env import GenesisEnv

RenderFrame = TypeVar("RenderFrame")


class Wrapper:
    """
    The core wrapper class that provides the basic functionality for all wrappers.
    """

    env: GenesisEnv = None

    def __init__(self, env: GenesisEnv):
        """Initialize the logging wrapper with the function to use for data logging."""
        self.env = env
        if not isinstance(env, GenesisEnv) and not isinstance(env, Wrapper):
            raise ValueError(
                f"Expected env to be a `GenesisEnv` or `Wrapper` but got {type(env)}"
            )

    @property
    def dt(self) -> float:
        """The time step of the environment."""
        return self.env.dt

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self.env.num_envs

    @property
    def scene(self) -> gs.Scene:
        """Get the environment scene."""
        return self.env.scene

    @property
    def robot(self) -> Any:
        """Get the environment robot."""
        return self.env.robot

    @property
    def action_space(self) -> spaces:
        """The action space of the environment."""
        return self.env.action_space

    @property
    def observation_space(self) -> spaces:
        """The observation space of the environment."""
        return self.env.observation_space

    def construct_scene(self) -> gs.Scene:
        """Uses the :meth:`construct_scene` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.construct_scene()

    def build_scene(self) -> None:
        """Builds the scene once all entities have been added (via construct_scene). This operation is required before running the simulation."""
        if self.env.scene is None:
            self.construct_scene()
        self.env.build_scene()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.step(actions)

    def reset(
        self,
        env_ids: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.reset(env_ids)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self.env.close()

    def set_data_tracker(self, _track_data_fn: Callable[[str, float], None]):
        """Set the function which logs data to tensorboard."""
        self.env.set_data_tracker(_track_data_fn)

    def track_data(self, name: str, value: float):
        """Log a single value to tensorboard, or similar"""
        self.env.track_data(name, value)
