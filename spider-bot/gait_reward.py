import torch
import genesis as gs
from typing import TYPE_CHECKING
from genesis_forge.managers import ContactManager, MdpFnClass
from genesis_forge.genesis_env import GenesisEnv


if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class GaitReward(MdpFnClass):
    """
    Reward foot groups that have at least half of the feet in contact.
    This is a very basic way to reward stable walking and encourage an alternating tetrapod gait.

    It rewards on a minimum number of contacts (50% of the feet in the group), and the robot is not penalized for having all
    feet on the ground (for example, when standing still). But it also does not give additional rewards when more than half
    of the feet are in contact, to avoid encouraging static standing. This allows the robot to learn optimal gait timing for
    different velocities and terrain.

    Args:
        env: The environment
        foot_groups: The foot groups to reward. Each group is a list of link names,
                     and the reward is based on whether at least half of the feet in each group are in contact.
        contact_manager: The contact manager for the feet
        force_threshold: The threshold for contact force to be considered in contact
        entity_attr: The attribute name of the entity in the environment

    Returns:
        The reward for the feet being synchronized in stable stance (0.0 to 1.0)

    Example::
        self.reward_manager = RewardManager(
            self,
            cfg={
                "foot_sync": {
                    "weight": 0.25,
                    "fn": GaitReward,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "foot_groups": [
                            ["FL_Foot", "RR_Foot"],
                            ["FR_Foot", "RL_Foot"],
                        ],
                    },
                },
            },
        )
    """

    def __init__(
        self,
        env: GenesisEnv,
        foot_groups: list[list[str]],
        contact_manager: ContactManager,
        force_threshold: float = 1.0,
        entity_attr: str = "robot",
    ):
        super().__init__(env)

        self._foot_group_names = foot_groups
        self._foot_group_link_idx = []
        self._robot: RigidEntity = getattr(env, entity_attr)
        self._group_rewards = torch.zeros(
            (self.env.num_envs, len(foot_groups)), device=gs.device
        )

    def build(self):
        """
        Get foot link indices
        """
        super().build()

        # Organize feet into the movement groups
        self._foot_group_link_idx = []
        for group in self._foot_group_names:
            group_link_idx = []
            for link_name in group:
                link = self._robot.get_link(link_name)
                if link is None:
                    raise ValueError(f"Foot link {link_name} not found")
                group_link_idx.append(link.idx)
            self._foot_group_link_idx.append(group_link_idx)

    def __call__(
        self,
        env: GenesisEnv,
        contact_manager: ContactManager,
        force_threshold: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        for i, group in enumerate(self._foot_group_link_idx):
            # Get contact state for all feet in group
            contact_forces = contact_manager.get_contact_forces(group).norm(dim=-1)
            num_in_contact = (contact_forces >= force_threshold).float().sum(dim=-1)

            num_feet = len(group)
            feet_multiplier = 1 / num_feet
            stable_contact = num_in_contact >= (num_feet / 2)

            # Calculate sync reward (0.0 to 1.0)
            # The reward is the same if half or more feet are in contact
            sync_score = torch.clamp(num_in_contact * feet_multiplier, max=0.5) * 2.0
            self._group_rewards[:, i] = sync_score * stable_contact.float()

        return self._group_rewards.mean(dim=-1)
