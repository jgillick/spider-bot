"""Spider locomotion configuration package."""

from .spider_env_cfg import SpiderLocomotionEnvCfg, SpiderSceneCfg
from .spider_flat_env_cfg import SpiderLocomotionFlatEnvCfg

__all__ = [
    "SpiderSceneCfg",
    "SpiderLocomotionEnvCfg",
    "SpiderLocomotionFlatEnvCfg",
]
