# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

from . import agents


##
# Register Gym environments.
##

gym.register(
    id="SpiderBot-Flat-v0",
    entry_point=f"{__name__}.managers:SpiderBotRLManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spider_env_cfg:SpiderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpiderBotPPORunnerCfg",
    },
)

