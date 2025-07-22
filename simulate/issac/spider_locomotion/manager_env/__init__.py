"""Spider locomotion configuration package."""

from .spider_bot_cfg import SpiderBotCfg
from .spider_env_cfg import SpiderLocomotionEnvCfg, SpiderSceneCfg
from .spider_flat_env_cfg import SpiderLocomotionFlatEnvCfg

__all__ = [
    "SpiderBotCfg",
    "SpiderSceneCfg",
    "SpiderLocomotionEnvCfg",
    "SpiderLocomotionFlatEnvCfg",
]
