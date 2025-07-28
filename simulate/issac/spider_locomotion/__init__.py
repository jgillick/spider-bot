"""Spider locomotion environment for Isaac Lab."""

import gymnasium as gym
from . import agents

# Register for rough terrain
gym.register(
    id="Isaac-SpiderLocomotion-v0",
    entry_point=f"{__name__}.manager_env:SpiderBotRLManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manager_env.spider_env_cfg:SpiderLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotRoughPPORunnerCfg",
    },
)

# Register for flat terrain variant
gym.register(
    id="Isaac-SpiderLocomotion-Flat-v0",
    entry_point=f"{__name__}.manager_env:SpiderBotRLManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manager_env.spider_flat_env_cfg:SpiderLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotFlatPPORunnerCfg",
    },
)

# Register for direct workflow (manual, non-manager-based)
gym.register(
    id="Isaac-SpiderLocomotion-Direct-v0",
    entry_point=f"{__name__}.direct_env:SpiderLocomotionFlatDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:SpiderFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotFlatPPORunnerCfg",
    },
)
