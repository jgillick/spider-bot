"""Spider locomotion environment for Isaac Lab."""

import gymnasium as gym

from .config.spider_env_cfg import SpiderLocomotionEnvCfg
from .config.spider_flat_env_cfg import SpiderLocomotionFlatEnvCfg

# Register the environment with Gymnasium
gym.register(
    id="Isaac-SpiderLocomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SpiderLocomotionEnvCfg,
    },
)

# Register for flat terrain variant
gym.register(
    id="Isaac-SpiderLocomotion-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SpiderLocomotionFlatEnvCfg,
    },
)
