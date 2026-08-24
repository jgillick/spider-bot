import torch
import genesis as gs

from genesis_forge import GenesisEnv
from genesis_forge.managers import ResetMdpFnClass, TerrainManager
from genesis_forge.utils import links_by_name_pattern

from genesis.engine.entities import RigidEntity


class RandomizeFrictionRatio(ResetMdpFnClass):
    """
    Randomize the friction of one or more links, per environment, on every reset.

    A random friction ratio (multiplier on the geom's base friction coefficient) is
    sampled uniformly from `friction_range` for each link and environment being reset.

    Genesis resolves the friction of a contact pair as the *maximum* of the two geoms'
    coefficients, so at build time the terrain's friction is set to the bottom of
    `friction_range` (clamped to Genesis's minimum stable value of 0.01). Otherwise
    a terrain with higher friction would silently clip the low end of the range.

    Args:
        env: The environment.
        entity: The entity to randomize link friction for (provided by EntityManager).
        terrain_manager: The terrain manager for the terrain the entity walks on.
        link_names: The names, or regex patterns, of the link(s) to randomize.
        friction_range: (min, max) friction ratio sampled per link per reset.

    Example::
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "friction": {
                    "fn": RandomizeFrictionRatio,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                        "link_names": ["[RL][1-4]_Foot"],
                        "friction_range": (0.4, 1.8),
                    },
                },
            },
        )
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        terrain_manager: TerrainManager,
        link_names: list[str],
        friction_range: tuple[float, float],
    ):
        super().__init__(env, entity)
        self.env = env
        self._entity = entity
        self._terrain_manager = terrain_manager
        self._link_names = link_names
        self._friction_range = friction_range
        self._links_idx_local: list[int] = []

    def build(self):
        self._links_idx_local = []
        for pattern in self._link_names:
            links = links_by_name_pattern(self._entity, pattern)
            if len(links) == 0:
                raise ValueError(f"No links found with name/pattern '{pattern}'")
            self._links_idx_local.extend(link.idx_local for link in links)
        # De-duplicate while preserving order, in case patterns overlap
        self._links_idx_local = list(set(self._links_idx_local))

        # Keep the terrain at/below the bottom of the range so the max-rule
        # never clips the randomization (TerrainManager has no public entity
        # accessor, so resolve it from the env the same way the manager does).
        terrain: RigidEntity = getattr(self.env, self._terrain_manager._terrain_attr)
        terrain.set_friction(max(0.01, self._friction_range[0]))

    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
        **params,
    ):
        ratio = torch.empty(
            (len(envs_idx), len(self._links_idx_local)), device=gs.device
        ).uniform_(*self._friction_range)
        self._entity.set_friction_ratio(
            ratio,
            links_idx_local=self._links_idx_local,
            envs_idx=envs_idx,
        )
