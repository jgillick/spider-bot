import torch
import genesis as gs
from genesis_forge import GenesisEnv
from genesis.utils.geom import transform_by_quat, inv_quat
from genesis_forge.utils import links_by_name_pattern

class FootAngleMdp:
    """
    A class that tracks the angle of the tibia, and provides penalties and termination conditions
    if the leg is angled under the body.
    The penalty is the sum of how far the projected gravity of each leg is below zero.

    Args:
        env: The environment to penalize.
        target_angle: The target foot angle. Anything less than this will be penalized.
    """
    def __init__(self, env: GenesisEnv):
        self.env = env
    
    def build(self):
        # Get foot link indices
        self._foot_links_idx = []
        for link in links_by_name_pattern(self.env.robot, "Leg[1-8]_Tibia_Foot"):
            self._foot_links_idx.append(link.idx_local)

        # Buffers
        self._foot_link_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self._foot_link_gravity = self._foot_link_gravity.unsqueeze(0).expand(
            self.env.num_envs, len(self._foot_links_idx), 3
        )
        self.step_leg_angle = torch.zeros(self.env.num_envs, len(self._foot_links_idx), device=gs.device)
    
    def calculate_leg_angle(self):
        quats = self.env.robot.get_links_quat(links_idx_local=self._foot_links_idx)

        # Transform link frames to world gravity frames
        inv_quats = inv_quat(quats)
        gravity_in_links = transform_by_quat(self._foot_link_gravity, inv_quats)
        self.step_leg_angle = -gravity_in_links[..., 2]
    
    def terminate(self, env: GenesisEnv, angle_threshold: float = -0.75):
        """
        Terminate if any leg is angled under the body by more than the threshold.
        """
        self.calculate_leg_angle()
        return torch.sum(self.step_leg_angle <= angle_threshold, dim=-1) > 0

    def reward(self, env: GenesisEnv, target_angle: float = 0.0):
        """
        Calculate the reward for the leg angle.
        """
        angles = self.step_leg_angle
        return torch.abs(torch.sum(angles * (angles < target_angle), dim=1))
