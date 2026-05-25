import torch
import genesis as gs
from typing import TYPE_CHECKING, TypedDict
from genesis_forge.managers.command.velocity_command import (
    VelocityCommandManager,
    VelocityDebugVisualizerConfig,
    DEFAULT_VISUALIZER_CONFIG,
    VelocityCommandRange,
)
from genesis_forge.managers import EntityManager, ContactManager, TerrainManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

VELOCITY_MAX_X = 1.2
VELOCITY_MAX_Y = 1.0


class FootNames(TypedDict):
    L1: str
    L2: str
    L3: str
    L4: str
    R1: str
    R2: str
    R3: str
    R4: str


class GaitCommandManager(VelocityCommandManager):

    # Create diagonal foot pairs that should alternate in steps
    foot_group_cfg = [
        ["L1", "R2"],
        ["L2", "R1"],
        ["L3", "R4"],
        ["L4", "R3"],
    ]

    def __init__(
        self,
        env: GenesisEnv,
        foot_names: FootNames,
        velocity_range: VelocityCommandRange,
        resample_time_sec: float = 5.0,
        jumping_probability: float = 0.1,
        standing_probability: float = 0.1,
        debug_visualizer: bool = False,
        entity_attr: str = "robot",
        debug_visualizer_cfg: VelocityDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(
            env,
            range=velocity_range,
            standing_probability=standing_probability,
            resample_time_sec=resample_time_sec,
            debug_visualizer=debug_visualizer,
            debug_visualizer_cfg=debug_visualizer_cfg,
        )

        self._gamepad: Gamepad | None = None
        self._gamepad_btn_pressed: bool = False
        self._gamepad_gait_idx = 0
        self._foot_names = foot_names
        self._foot_groups = []
        self._foot_link_idx = []
        self.jumping_probability = jumping_probability
        self._robot:RigidEntity  = getattr(env, entity_attr)

        # Buffers
        self._jumping_envs = torch.zeros(env.num_envs, device=gs.device, dtype=torch.bool)

    """
    Curriculum operations
    """

    def increase_velocity(self):
        """
        If training is going well, increase the possible velocity ranges by 0.05.
        """
        vel_range = self.range
        vel_range["lin_vel_x"][0] = max(
            vel_range["lin_vel_x"][0] - 0.05, -VELOCITY_MAX_X
        )
        vel_range["lin_vel_x"][1] = min(
            vel_range["lin_vel_x"][1] + 0.05, VELOCITY_MAX_X
        )

        vel_range["lin_vel_y"][0] = max(
            vel_range["lin_vel_y"][0] - 0.05, -VELOCITY_MAX_Y
        )
        vel_range["lin_vel_y"][1] = min(
            vel_range["lin_vel_y"][1] + 0.05, VELOCITY_MAX_X
        )

    """
    Command lifecycle operations
    """

    def resample_command(self, env_ids: list[int]):
        """
        Resample the command for the given environments
        """
        # Do not resample if using gamepad control
        if self._gamepad is not None:
            return

        # Sample velocity command
        super().resample_command(env_ids)

        # Add jumping environments
        jump_envs = torch.empty(len(env_ids), device=gs.device).uniform_(0.0, 1.0)
        self._jumping_envs[env_ids] = (jump_envs <= self.jumping_probability)

    def build(self):
        """
        Get foot link indices
        """
        super().build()

        # Organize feet into the movement groups
        self._foot_groups = []
        for key, link_name in self._foot_names.items():
            link = self._robot.get_link(link_name)
            self._foot_link_idx.append(link.idx_local)
            for g, group in enumerate(self.foot_group_cfg):
                if len(self._foot_groups) < g + 1:
                    self._foot_groups.append([])
                if key in group:
                    self._foot_groups[g].append(link.idx)

    def observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        Return command observations
        """
        return torch.cat(
            [
                self.command,  # velocity command
                self._jumping_envs.unsqueeze(-1),
            ],
            dim=-1,
        )

    def privileged_observation(
        self, env: GenesisEnv, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Return priviledged observations
        """
        return torch.norm(contact_manager.contacts[:, :, :], dim=-1)

    def use_gamepad(self, gamepad: Gamepad):
        """
        Control the command using a gamepad.
        Pressing the A button will cycle through the gaits.
        """
        pass

    """
    Rewards
    """

    def foot_sync_reward(
        self,
        env: GenesisEnv,
        contact_manager: ContactManager,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        """
        Reward synchronization within foot groups when in stable contact.

        For each group, rewards when at least half of feet are in contact (stable stance).
        The reward maxes out at 50% contact to avoid encouraging static standing with
        all feet on the ground, while still rewarding stable gaits (2 or 4 feet down).
        This allows the robot to learn optimal gait timing for different velocities and terrain.

        Args:
            env: The environment
            contact_manager: The contact manager
            force_threshold: The threshold for contact force to be considered in contact

        Returns:
            The reward for the feet being synchronized in stable stance (0.0 to 1.0)
        """
        group_rewards = torch.zeros(
            (self.env.num_envs, len(self._foot_groups)), device=gs.device
        )
        for i, group in enumerate(self._foot_groups):
            # Get contact state for all feet in group
            contact_forces = contact_manager.get_contact_forces(group).norm(dim=-1)
            num_in_contact = (contact_forces >= force_threshold).float().sum(dim=-1)

            num_feet = len(group)
            feet_multiplier = 1 / num_feet
            stable_contact = num_in_contact >= (num_feet / 2)

            # Calculate sync reward (0.0 to 1.0)
            # The reward is the same if half or more feet are in contact
            sync_score = torch.clamp(num_in_contact * feet_multiplier, max=0.5) * 2.0
            group_rewards[:, i] = sync_score * stable_contact.float()

        return group_rewards.mean(dim=-1) * ~self._jumping_envs

    def jump_reward(
        self,
        env: GenesisEnv,
        terrain_manager: TerrainManager,
        entity_manager: EntityManager,
        foot_height_threshold: float = 0.05,
    ) -> torch.Tensor:
        """
        Calculate the reward for the robot jumping.
        Rewards:
        - Vertical velocity (Z-axis)
        - Forward velocity (X-axis)
        - Number of feet off the ground

        Args:
            env: The environment
            terrain_manager: The terrain manager, use to calculate the height of the feet off the ground
            entity_manager: The entity manager
            foot_height_threshold: Minimum height for a foot to be considered off the ground (default 0.05m)

        Returns:
            Combined reward tensor of shape (num_envs,)
        """
        # Get base velocity of the robot
        base_vel = entity_manager.get_linear_velocity()
        vel_x = base_vel[:, 0]  # Forward velocity
        vel_z = base_vel[:, 2]  # Vertical velocity

        # Count feet off the ground (all 8 feet)
        foot_pos = self._robot.get_links_pos(links_idx_local=self._foot_link_idx)
        for i in range(8):
            terrain_height = terrain_manager.get_terrain_height(
                foot_pos[:, i, 0], foot_pos[:, i, 1]
            )
            foot_pos[:, i, 2] += terrain_height
        foot_height = foot_pos[:, :, 2]
        feet_airborne = (foot_height > foot_height_threshold).float()
        num_feet_airborne = feet_airborne.sum(dim=1)

        # Combine rewards
        # Reward positive vertical velocity more heavily, forward velocity moderately, and feet off ground
        vertical_reward = torch.clamp(vel_z, min=0.0)  # Only reward upward velocity
        forward_reward = torch.clamp(vel_x, min=0.0)  # Only reward forward velocity
        feet_reward = num_feet_airborne / 8.0  # Normalize to 0-1 range

        # Combined reward with weights
        # Emphasize vertical velocity and feet off ground for jumping
        reward = (
            2.0 * vertical_reward  # Strong emphasis on vertical velocity
            + 1.0 * forward_reward  # Moderate emphasis on forward velocity
            + 3.0 * feet_reward  # Strong emphasis on getting all feet off ground
        )

        return reward * self._jumping_envs

