from typing import Sequence

from genesis_forge.genesis_env import GenesisEnv


class BaseManager:
    """
    A base manager describing the interface for all other managers
    """

    env: GenesisEnv

    def __init__(
        self,
        env: GenesisEnv,
    ):
        self.env = env

    def construct_scene(self):
        """Called when the scene is constructed"""
        pass

    def step(self):
        """Called when the environment is stepped"""
        pass

    def reset(self, env_ids: Sequence[int] = None):
        """One or more environments have been reset"""
        pass
