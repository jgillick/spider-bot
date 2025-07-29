
import torch
import gymnasium as gym
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn

from .metric_manager import MetricManager

class SpiderBotRLManagerEnv(ManagerBasedRLEnv):
    """
    RL Manager that limits the action space to be -1 to 1 and adds metrics to the logger
    """
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        print("SPIDERBOT RL MANAGER ENV")
        super().__init__(cfg, render_mode, **kwargs)

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()

        # Set action space
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        ret = super().step(action)
        self.metrics_manager.compute()
        return ret
    
    def load_managers(self):
        super().load_managers()
        self.metrics_manager = MetricManager(self.cfg.metrics, self)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        metric_logs = self.metrics_manager.reset(env_ids)
        self.extras["log"].update(metric_logs)
    
    def close(self):
        if not self._is_closed:
            del self.metrics_manager
            super().close()