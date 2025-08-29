import torch
import genesis as gs
from typing import Any, Tuple, Callable, Union, Sequence

from genesis_forge.wrappers.wrapper import Wrapper
from genesis_forge.genesis_env import GenesisEnv

LogFn = Union[Callable[[str, float], None], None]


class DataLoggerWrapper(Wrapper):
    """
    Wrapper that automatically logs data from the info dict to tensorboard or other logging systems.

    Args:
        env: The environment to wrap.
        logger_fn: A function that will be called with the name and value of each logged value.
        log_key: The key in the info dict that contains the items to log. (default: "logs")

    Example::
        class MyEnv(GenesisEnv):
            def step(self, actions):
                # ... Do some step stuff

                # Initialize the logs dict
                info["logs"] = {}

                # Step logs
                info["logs"] = {}
                info["logs"]["Section / My Chart"] = Tensor.zeros((num_envs,))
                info["logs"]["Section / My Chart"][0] = 1.0

                # Return
                return obs, rewards, terminations, timeouts, info
        
        def train():
            # Create wrapped environment
            env = MyEnv()
            env = DataLoggerWrapper(env)
            env.build()

            # Setup SKRL training runner
            skrl_env = create_skrl_env(env)
            runner = Runner(skrl_env, cfg)

            # Assign the SKRL data tracker to the environment
            env.set_data_tracker(runner.agent.track_data)

            # Train
            runner.run("train")
    """

    key: str
    log_fn: LogFn = None

    def __init__(self, env: GenesisEnv, logger_fn: LogFn = None, log_key: str = "logs"):
        """Initialize the logging wrapper with the function to use for data logging."""
        super().__init__(env)
        self.key = log_key
        self.log_fn = logger_fn

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment and log data."""
        obs, rewards, terminations, timeouts, info = super().step(actions)
        if self.key in info:
            self._log_items(info[self.key])
        return obs, rewards, terminations, timeouts, info

    def reset(
        self, env_ids: Sequence[int] = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        (obs, info) = super().reset(env_ids)
        if self.key in info:
            self._log_items(info[self.key])
        return obs, info
    
    def _log_items(self, logs: dict[str, Any]):
        """Log items from the info dict."""
        for name, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.float().mean().cpu().item()
            self.track_data(name, value)
