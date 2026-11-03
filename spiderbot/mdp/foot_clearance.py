import torch
from dataclasses import dataclass
from genesis_forge import GenesisEnv
from genesis_forge.managers import ContactManager, TerrainManager
from genesis_forge.managers import MdpFn


@dataclass(kw_only=True, eq=False)
class FootClearanceMdp(MdpFn):
    """
    Penalizes swinging feet that do not reach a minimum clearance height above the terrain.

    The penalty is applied only during mid-swing (after the foot has been in the air longer
    than `liftoff_margin`) to avoid penalizing the brief liftoff and touchdown transitions
    where the foot is technically in swing but still close to the ground.

    The penalty scales with the squared deficit below `target_clearance`, providing a strong
    gradient when feet barely lift off and a soft signal when close to the target.

    Args:
        contact_manager: Contact manager for the feet (provides link indices and air time).
        terrain_manager: Used to sample terrain height at each foot's xy position.
        target_clearance: Minimum foot height above terrain (in metres) to aim for during swing.
        liftoff_margin: Seconds the foot must have been airborne before the penalty activates.
                        Skips the initial liftoff steps where the foot is still near the ground.

    Example::
        self.reward_manager = RewardManager(
            self,
            cfg={
                "foot_clearance": {
                    "weight": -0.1,
                    "fn": FootClearanceMdp(
                        contact_manager=self.foot_contact_manager,
                        terrain_manager=self.terrain_manager,
                    ),
                },
            },
        )
    """

    contact_manager: ContactManager | None = None
    terrain_manager: TerrainManager | None = None
    target_clearance: float = 0.05
    liftoff_margin: float = 0.1

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        """
        Returns the per-environment clearance penalty (summed over all feet).
        Should be registered with a negative weight.
        """
        foot_link_ids_local = self.contact_manager.local_link_ids
        n_feet = foot_link_ids_local.shape[0]

        # Foot world positions: (n_envs, n_feet, 3)
        foot_pos = env.robot.get_links_pos(links_idx_local=foot_link_ids_local)

        # Query terrain height per foot. get_terrain_height expects (n_envs,) tensors,
        # so we iterate over feet rather than flattening all at once.
        terrain_z = torch.stack(
            [
                self.terrain_manager.get_terrain_height(
                    foot_pos[:, i, 0], foot_pos[:, i, 1]
                )
                for i in range(n_feet)
            ],
            dim=1,
        )  # (n_envs, n_feet)

        # Clearance above terrain for each foot
        clearance = foot_pos[..., 2] - terrain_z

        # Only penalize during mid-swing (airborne longer than liftoff_margin)
        mid_swing = self.contact_manager.current_air_time > self.liftoff_margin

        # Squared deficit below target clearance, zero when clearance meets target
        deficit = torch.clamp(self.target_clearance - clearance, min=0.0)
        penalty = (deficit**2) * mid_swing

        return penalty.sum(dim=-1)
