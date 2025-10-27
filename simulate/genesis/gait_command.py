"""
Gait Command Manager for implementing the periodic reward composition method
from "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
"""

import torch
import genesis as gs
from typing import TYPE_CHECKING, TypedDict, Literal
from genesis_forge.managers.command.velocity_command import (
    VelocityCommandManager,
    VelocityDebugVisualizerConfig,
    DEFAULT_VISUALIZER_CONFIG,
)
from genesis_forge.managers import EntityManager, TerrainManager, ContactManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

GAIT_PERIOD_RANGE = [0.3, 0.4]
GAIT_PERIOD_MAX = 0.6
FOOT_CLEARANCE_RANGE = [0.1, 0.15]

FootName = Literal["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

STANCE_PHASE = 0.75


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
    """
    Manages parameters/rewards for different locomotion gaits.
    Based on the paper "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition" (Siekmann et al., 2020)
    https://arxiv.org/abs/2011.01387
    """

    foot_group_cfg = [
        ["L1", "L3", "R2", "R4"],
        ["L2", "L4", "R1", "R3"],
    ]

    gaits = [
        {
            "name": "walk",  # Alternating Tetrapod
            "offsets": {
                "L1": 0.0,
                "L2": 0.5,
                "L3": 0.0,
                "L4": 0.5,
                "R1": 0.5,
                "R2": 0.0,
                "R3": 0.5,
                "R4": 0.0,
            },
        },
        {
            "name": "bound",
            "offsets": {
                "L1": 0.0,
                "L2": 0.5,
                "L3": 0.0,
                "L4": 0.5,
                "R1": 0.0,
                "R2": 0.5,
                "R3": 0.0,
                "R4": 0.5,
            },
        },
        {
            "name": "pronk",
            "offsets": {
                "L1": 0.0,
                "L2": 0.0,
                "L3": 0.0,
                "L4": 0.0,
                "R1": 0.0,
                "R2": 0.0,
                "R3": 0.0,
                "R4": 0.0,
            },
        },
    ]

    def __init__(
        self,
        env: GenesisEnv,
        foot_names: FootNames,
        entity_manager: EntityManager,
        terrain_manager: TerrainManager | None = None,
        resample_time_sec: float = 5.0,
        robot_entity_attr: str = "robot",
        debug_visualizer: bool = False,
        debug_visualizer_cfg: VelocityDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(
            env,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.0,
            resample_time_sec=resample_time_sec,
            debug_visualizer=debug_visualizer,
            debug_visualizer_cfg=debug_visualizer_cfg,
        )

        self._robot: RigidEntity | None = None
        self._robot_entity_attr = robot_entity_attr
        self._foot_names = foot_names
        self._gamepad: Gamepad | None = None
        self._gamepad_btn_pressed: bool = False
        self._gamepad_gait_idx = 0
        self._foot_links = []
        self._foot_link_idx = []
        self._entity_mgr = entity_manager
        self._terrain_mgr = terrain_manager
        self._gait_period_range = [0.4, 0.4]
        self._foot_groups = []

        # Initial ranges - these will be expanded in the curriculum
        self.num_gaits = 1
        self._foot_clearance_range = [FOOT_CLEARANCE_RANGE[0]] * 2
        self._all_gaits_learned = False

        # Buffers
        self._foot_offset = torch.zeros((env.num_envs, 8), device=gs.device)
        self._gait_period = torch.zeros((env.num_envs, 1), device=gs.device)
        self._target_foot_height = torch.zeros((env.num_envs, 1), device=gs.device)
        self._gait_time = torch.zeros(
            env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        self._gait_phase = torch.zeros(
            env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        self._clock_input = torch.zeros(
            env.num_envs,
            16,
            dtype=torch.float,
            device=gs.device,
        )
        self._gait_selected = torch.zeros(
            env.num_envs, dtype=torch.long, device=gs.device
        )
        self._curr_foot_height = torch.zeros((env.num_envs, 8), device=gs.device)
        self._velocity_command = torch.zeros((env.num_envs, 3), device=gs.device)

    """
    Properties
    """

    @property
    def gait_command(self) -> torch.Tensor:
        """
        Return the selected gait.
        """
        return torch.cat(
            [
                self._gait_selected.unsqueeze(-1),
                self._foot_offset,
                self._target_foot_height,
                self._gait_period,
            ],
            dim=-1,
        )

    """
    Curriculum operations
    """

    def increment_num_gaits(self):
        """
        If training is going well, increase the number of available gaits by 1.
        """
        if self._all_gaits_learned:
            return

        # All gaits have now been mastered
        if self.num_gaits == len(self.gaits):
            self._all_gaits_learned = True
            print("🎯 All gaits learned! Switching to uniform sampling.")
        else:
            self.num_gaits = min(self.num_gaits + 1, len(self.gaits))

    def increment_gait_period_range(self):
        """
        If training is going well, increase the possible gait period range by 0.05.
        """
        self._gait_period_range[1] = min(
            self._gait_period_range[1] + 0.05, GAIT_PERIOD_RANGE[1]
        )

    def increment_gait_speeds(self):
        """
        If training is going well, increase the possible gait speeds by 0.05.
        """
        vel_range = self.range
        vel_range["lin_vel_x"][0] = max(vel_range["lin_vel_x"][0] - 0.05, -1.2)
        vel_range["lin_vel_x"][1] = min(vel_range["lin_vel_x"][1] + 0.05, 1.2)
        # vel_range["lin_vel_y"][0] = max(vel_range["lin_vel_y"][0] - 0.05, -1.0)
        # vel_range["lin_vel_y"][1] = min(vel_range["lin_vel_y"][1] + 0.05, 1.0)

    def increment_foot_clearance_range(self):
        """
        If training is going well, increase the possible foot clearance range by 0.05.
        """
        self._foot_clearance_range[0] = max(
            self._foot_clearance_range[0] - 0.01, FOOT_CLEARANCE_RANGE[0]
        )
        self._foot_clearance_range[1] = min(
            self._foot_clearance_range[1] + 0.01, FOOT_CLEARANCE_RANGE[1]
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

        # Convert env_ids to tensor if it's a list
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=gs.device, dtype=torch.long)

        if self.num_gaits == 1:
            # Only one gait available - set all to the same gait
            self._set_gait(0, env_ids)
        else:
            # Generate a random list of gait indices
            gait_indices = self._generate_random_gait_indices(len(env_ids))
            for gait_idx in range(self.num_gaits):
                mask = gait_indices == gait_idx
                if mask.any():
                    selected_envs = env_ids[mask]
                    self._set_gait(gait_idx, selected_envs)

    def build(self):
        """
        Get foot link indices
        """
        super().build()
        self._robot: RigidEntity = getattr(self.env, self._robot_entity_attr)
        self._foot_groups = [[], []]
        for i, key in enumerate(self._foot_names.keys()):
            foot_link_name = self._foot_names[key]
            link = self._robot.get_link(foot_link_name)
            self._foot_links.insert(i, link)
            self._foot_link_idx.insert(i, link.idx_local)

            # Add to group
            for g, group in enumerate(self.foot_group_cfg):
                if key in group:
                    self._foot_groups[g].append(link.idx)

    def step(self):
        """
        Increment the gait time and phase
        """
        super().step()

        # Get foot height off the terrain
        foot_pos = self._robot.get_links_pos(links_idx_local=self._foot_link_idx)
        if self._terrain_mgr is not None:
            for i in range(8):
                terrain_height = self._terrain_mgr.get_terrain_height(
                    foot_pos[:, i, 0], foot_pos[:, i, 1]
                )
                foot_pos[:, i, 2] += terrain_height
        self._curr_foot_height = foot_pos[:, :, 2]

        # Metrics
        self._log_metrics()

        # Update periodic gait values
        self._gait_time = (self._gait_time + self.env.dt) % self._gait_period
        self._gait_phase = self._gait_time / self._gait_period
        for i in range(8):  # for each leg
            foot_phase = (self._gait_phase + self._foot_offset[:, i].unsqueeze(1)) % 1.0
            self._clock_input[:, i] = torch.sin(2 * torch.pi * foot_phase).squeeze(-1)
            self._clock_input[:, i + 8] = torch.cos(2 * torch.pi * foot_phase).squeeze(
                -1
            )

    def reset(self, env_ids: list[int] | None = None):
        """
        Reset environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        super().reset(env_ids)
        self._clock_input[env_ids, :] = 0.0
        self._gait_time[env_ids] = 0.0
        self._gait_phase[env_ids] = 0.0

    def observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        Return command observations
        """
        return torch.cat(
            [
                self.command,
                # self.gait_command,
                # self._clock_input,
            ],
            dim=-1,
        )

    def privileged_observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        Return private command observations
        """
        return torch.cat(
            [
                self._curr_foot_height,
            ],
        )

    def use_gamepad(self, gamepad: Gamepad):
        """
        Control the command using a gamepad.
        Pressing the A button will cycle through the gaits.
        """
        pass

    """
    Rewards
    """

    def foot_height_reward(
        self, env: GenesisEnv, sensitivity: float = 0.1
    ) -> torch.Tensor:
        """
        Calculate the reward for the feet reaching the target height during the swing phase
        """
        foot_vel = env.robot.get_links_vel(links_idx_local=self._foot_link_idx)
        foot_vel_xy_norm = torch.norm(foot_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(self._curr_foot_height - self._target_foot_height),
            dim=-1,
        )
        is_moving = self.command.norm(dim=-1) > 0.02
        return torch.exp(-clearance_error / sensitivity) * is_moving

    def gait_phase_reward(
        self, env: GenesisEnv, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the reward for the feet being in the correct phase.
        """
        reward = torch.zeros(self.env.num_envs, device=gs.device)
        for i in range(8):  # for each leg
            reward += self._foot_phase_reward(i, contact_manager).flatten()
        is_moving = self.command.norm(dim=-1) > 0.02
        return torch.exp(reward) * is_moving
    
    def foot_sync_reward(self, env: GenesisEnv, contact_manager: ContactManager, force_threshold: float = 1.0) -> torch.Tensor:
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
        group_rewards = torch.zeros((self.env.num_envs, len(self._foot_groups)), device=gs.device)
        for i, group in enumerate(self._foot_groups):
            # Get contact state for all feet in group
            contact_forces = contact_manager.get_contact_forces(group).norm(dim=-1)
            num_in_contact = (contact_forces >= force_threshold).float().sum(dim=-1)
            
            num_feet = len(group)
            feet_multiplier = 1/num_feet
            stable_contact = num_in_contact >= (num_feet / 2)
            
            # Calculate sync reward (0.0 to 1.0)
            # The reward is the same if half or more feet are in contact
            sync_score = torch.clamp(num_in_contact * feet_multiplier, max=0.5) * 2.0 
            group_rewards[:, i] = sync_score * stable_contact.float()
        
        return group_rewards.mean(dim=-1)


    def jump_reward(self, env: GenesisEnv, entity_manager: EntityManager, foot_height_threshold: float = 0.05) -> torch.Tensor:
        """
        Calculate the reward for the robot jumping.
        Rewards:
        - Vertical velocity (Z-axis)
        - Forward velocity (X-axis)
        - Number of feet off the ground
        
        Args:
            env: The environment
            foot_height_threshold: Minimum height for a foot to be considered off the ground (default 0.05m)
        
        Returns:
            Combined reward tensor of shape (num_envs,)
        """
        # Get base velocity of the robot
        base_vel = entity_manager.get_linear_velocity()
        vel_x = base_vel[:, 0]  # Forward velocity
        vel_z = base_vel[:, 2]  # Vertical velocity
        
        # Count feet off the ground (all 8 feet)
        feet_airborne = (self._curr_foot_height > foot_height_threshold).float()  
        num_feet_airborne = feet_airborne.sum(dim=1)  
        
        # Combine rewards
        # Reward positive vertical velocity more heavily, forward velocity moderately, and feet off ground
        vertical_reward = torch.clamp(vel_z, min=0.0)  # Only reward upward velocity
        forward_reward = torch.clamp(vel_x, min=0.0)   # Only reward forward velocity
        feet_reward = num_feet_airborne / 8.0  # Normalize to 0-1 range
        
        # Combined reward with weights
        # Emphasize vertical velocity and feet off ground for jumping
        reward = (
            2.0 * vertical_reward +     # Strong emphasis on vertical velocity
            1.0 * forward_reward +       # Moderate emphasis on forward velocity
            3.0 * feet_reward            # Strong emphasis on getting all feet off ground
        )
        
        return reward
        

    """
    Private methods
    """

    def _foot_phase_reward(
        self, foot_idx: int, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the individual foot phase reward
        """
        force_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        vel_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )

        # Force / velocity
        link = self._foot_links[foot_idx]
        force = torch.norm(contact_manager.get_contact_forces(link.idx), dim=-1).view(
            -1, 1
        )
        velocity = torch.norm(link.get_vel(), dim=-1).view(-1, 1)

        # Current foot phase
        phase = (self._gait_phase + self._foot_offset[:, foot_idx].unsqueeze(1)) % 1.0

        # Phase is in range [0, 1]. Split 50/50 between swing and stance.
        # Swing: [0, 0.5), Stance: [0.5, 1.0]
        swing_indices = phase < 0.5
        stance_indices = phase >= 0.5

        # During swing: penalize ground contact force (want foot in air)
        force_weight[swing_indices] = -1.0
        vel_weight[swing_indices] = 0.0

        # During stance: penalize foot velocity (want foot planted on ground)
        force_weight[stance_indices] = 0.0
        vel_weight[stance_indices] = -1.0

        return vel_weight * velocity + force_weight * force

    def _set_gait(self, gait_idx: int, env_ids: torch.Tensor | None = None):
        """
        Set the gait for a batch of environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        n_envs = len(env_ids)
        self._gait_selected[env_ids] = gait_idx
        gait_cgf = self.gaits[gait_idx]
        gait_name = gait_cgf["name"]

        # Define the foot offsets for the selected gait
        offsets = torch.zeros((8), device=gs.device, dtype=torch.float)
        for i, foot in enumerate(["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]):
            offsets[i] = gait_cgf["offsets"][foot]
        self._foot_offset[env_ids] = offsets

        # Foot clearance
        self._target_foot_height[env_ids, 0] = torch.empty(
            n_envs, device=gs.device
        ).uniform_(*self._foot_clearance_range)

        # Gait period
        self._gait_period[env_ids, 0] = torch.empty(n_envs, device=gs.device).uniform_(
            *self._gait_period_range
        )

    def _generate_random_gait_indices(self, num: int) -> torch.Tensor:
        """
        Pick a list of random gait indices, with the most recent gaits having a higher probability of
        being picked.
        If self._all_gaits_learned is True, all gaits are equally weighted.
        """

        if not self._all_gaits_learned:
            # If we haven't learned all gaits yet, weight the recent gaits exponentially higher
            weights = torch.arange(self.num_gaits, device=gs.device).exp()
        else:
            # Otherwise, all gaits are equally weighted
            weights = torch.ones(self.num_gaits, device=gs.device)

        # Normalize: the sum of all weights should be 1
        weights /= weights.sum()
        weights = weights[: self.num_gaits].expand(num, -1)

        return torch.multinomial(weights, 1).squeeze(-1)

    def _log_metrics(self):
        self.env.extras[self.env.extras_logging_key][
            "Metrics / num_gaits"
        ] = self.num_gaits

        # Log gait distribution if multiple gaits are active
        for i, gait in enumerate(self.gaits):
            gait_name = gait["name"]
            count = (self._gait_selected == i).sum()
            self.env.extras[self.env.extras_logging_key][
                f"Metrics / gait_{gait_name}_envs"
            ] = count

        # Foot height
        self.env.extras[self.env.extras_logging_key]["Metrics / avg_foot_height"] = (
            self._curr_foot_height.mean(dim=-1).mean()
        )

        # Walking gait period
        self.env.extras[self.env.extras_logging_key]["Metrics / walking_gait_max"] = (
            self._gait_period_range[1]
        )
