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

    def __init__(self, env: GenesisEnv, foot_name_pattern: str):
        self.env = env
        self.foot_name_pattern = foot_name_pattern

    def build(self):
        # Get foot link indices
        self._foot_links_idx = []
        for link in links_by_name_pattern(self.env.robot, self.foot_name_pattern):
            self._foot_links_idx.append(link.idx_local)

        # Buffers
        self.step_leg_angle = torch.zeros(
            self.env.num_envs, len(self._foot_links_idx), device=gs.device
        )

    def calculate_leg_angle(self):
        quats = self.env.robot.get_links_quat(links_idx_local=self._foot_links_idx)
        self.step_leg_angle = calculate_leg_angles(quats)

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


def calculate_leg_angles(foot_quats: torch.Tensor) -> torch.Tensor:
    """
    Calculate how far each foot link is angled above (positive) or under (negative) the body.

    Args:
        foot_quats: World-frame quaternions of the foot links, shape (..., n_feet, 4).

    Returns:
        The leg angle of each foot, shape (..., n_feet).
    """
    gravity = torch.tensor([0.0, 0.0, -1.0], device=foot_quats.device)
    gravity = gravity.expand(*foot_quats.shape[:-1], 3)
    gravity_in_links = transform_by_quat(gravity, inv_quat(foot_quats))
    return -gravity_in_links[..., 2]
