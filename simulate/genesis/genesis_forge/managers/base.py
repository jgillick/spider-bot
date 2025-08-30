from genesis_forge.genesis_env import GenesisEnv


class BaseManager:
    """
    A base manager describing the interface for all other managers
    """

    env: GenesisEnv
    enabled: bool = True

    def __init__(
        self,
        env: GenesisEnv,
        enabled: bool = True,
    ):
        self.env = env
        self.enabled = True

    def build(self):
        """Called when the scene is built"""
        pass

    def step(self):
        """Called when the environment is stepped"""
        pass

    def reset(self, env_ids: list[int] | None = None):
        """One or more environments have been reset"""
        pass
