import torch
import genesis as gs
from genesis_forge import GenesisEnv
from genesis_forge.managers.config import MdpFnClass
from genesis.utils.geom import transform_by_quat, inv_quat

class foot_angle_penalty(MdpFnClass):
    """
    Penalize the tibia bending in too far and under the robot.
    The penalty is the sum of how far the projected gravity of each leg is below zero.

    Args:
        env: The environment to penalize.
        target_angle: The target foot angle. Anything less than this will be penalized.
    """
    def __init__(self, env: GenesisEnv, target_angle: float = 0.0):
        super().__init__(env)
    
    def build(self):
        # Get foot link indices
        self._foot_links_idx = []
        for link in self.env.robot.links:
            if link.name.endswith("_Tibia_Foot"):
                self._foot_links_idx.append(link.idx_local)

        # Buffers
        self._foot_link_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self._foot_link_gravity = self._foot_link_gravity.unsqueeze(0).expand(
            self.env.num_envs, len(self._foot_links_idx), 3
        )

    def __call__(self, env: GenesisEnv, target_angle: float = 0.0):
        quats = env.robot.get_links_quat(links_idx_local=self._foot_links_idx)

        # Transform link frames to world gravity frames
        inv_quats = inv_quat(quats)
        gravity_in_links = transform_by_quat(self._foot_link_gravity, inv_quats)
        uprightness = -gravity_in_links[..., 2]

        # Add up all the uprightness values less than zero
        return torch.abs(torch.sum(uprightness * (uprightness < target_angle), dim=1))
