"""Spider locomotion environment for Isaac Lab."""

import gymnasium as gym
from . import agents

# Register for rough terrain
gym.register(
    id="Isaac-SpiderLocomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.spider_env_cfg:SpiderLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotRoughPPORunnerCfg",
    },
)

# Register for flat terrain variant
gym.register(
    id="Isaac-SpiderLocomotion-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.spider_flat_env_cfg:SpiderLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotRoughPPORunnerCfg",
    },
)
