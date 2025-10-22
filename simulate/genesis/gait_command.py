"""
Gait Command Manager for implementing the periodic reward composition method
from "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
"""

import torch
import genesis as gs
from typing import TYPE_CHECKING, TypedDict, Literal
from genesis_forge.managers.command import (
    CommandManager,
    VelocityCommandManager,
)
from genesis_forge.managers import EntityManager, TerrainManager, ContactManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

GAIT_PERIOD_RANGE = [0.3, 0.6]
FOOT_CLEARANCE_RANGE = [0.1, 0.35]
CURRICULUM_CHECK_EVERY_STEPS = 500

GaitName = Literal["walk", "trot", "pronk", "pace", "bound", "canter"]
FootName = Literal["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]

# Gait configuration
# Phase offsets for each foot in different gaits (0.0 = start of cycle, 0.5 = mid-cycle)
# Each gait defines when each foot contacts the ground relative each other in the gait cycle.
GAIT_OFFSETS: dict[GaitName, dict[FootName, float]] = {
    "walk": {
        "L1": 0.0,
        "L2": 0.5,
        "L3": 0.0,
        "L4": 0.5,
        "R1": 0.5,
        "R2": 0.0,
        "R3": 0.5,
        "R4": 0.0,
    },
    "trot": {
        "L1": 0.0,
        "L2": 0.25,
        "L3": 0.5,
        "L4": 0.75,
        "R1": 0.25,
        "R2": 0.0,
        "R3": 0.75,
        "R4": 0.5,
    },
    "jump": {
        "L1": 0.0,
        "L2": 0.0,
        "L3": 0.0,
        "L4": 0.0,
        "R1": 0.0,
        "R2": 0.0,
        "R3": 0.0,
        "R4": 0.0,
    },
}


class FootNames(TypedDict):
    L1: str
    L2: str
    L3: str
    L4: str
    R1: str
    R2: str
    R3: str
    R4: str


class GaitCommandManager(CommandManager):
    """
    Manages parameters/rewards for different locomotion gaits.
    Based on the paper "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition" (Siekmann et al., 2020)
    https://arxiv.org/abs/2011.01387
    """

    def __init__(
        self,
        env: GenesisEnv,
        foot_names: FootNames,
        velocity_command_manager: VelocityCommandManager,
        entity_manager: EntityManager,
        terrain_manager: TerrainManager | None = None,
        resample_time_sec: float = 5.0,
        robot_entity_attr: str = "robot",
    ):
        super().__init__(env, range={}, resample_time_sec=resample_time_sec)

        self._robot: RigidEntity | None = None
        self._robot_entity_attr = robot_entity_attr
        self._foot_names = foot_names
        self._gamepad: Gamepad | None = None
        self._gamepad_btn_pressed: bool = False
        self._gamepad_gait_idx = 0
        self._foot_links = []
        self._foot_link_idx = []
        self._entity_mgr = entity_manager
        self._vel_command_mgr = velocity_command_manager
        self._terrain_mgr = terrain_manager

        # Initial ranges - these will be expanded in the curriculum
        self._num_gaits = 1
        self._gait_period_range = [
            (GAIT_PERIOD_RANGE[0] + GAIT_PERIOD_RANGE[1]) / 2
        ] * 2
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
        self._gait_idx = torch.zeros(
            (env.num_envs,), dtype=torch.float, device=gs.device
        )
        self._curr_foot_height = torch.zeros((env.num_envs, 8), device=gs.device)

    @property
    def command(self) -> torch.Tensor:
        """
        The combined gait command
        """
        if self._gamepad is not None:
            self._process_gamepad_input()

        is_moving = torch.norm(self._vel_command_mgr.command[:, :2], dim=-1) > 0.1
        is_moving = is_moving.unsqueeze(-1)
        return torch.cat(
            [
                self._gait_idx.unsqueeze(-1),
                self._foot_offset * is_moving,
                self._target_foot_height * is_moving,
                self._gait_period * is_moving,
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
        if self._num_gaits == len(GAIT_OFFSETS):
            self._all_gaits_learned = True
            print("🎯 All gaits learned! Switching to uniform sampling.")
        else:
            self._num_gaits = min(self._num_gaits + 1, len(GAIT_OFFSETS))

    def increment_gait_period_range(self):
        """
        If training is going well, increase the possible gait period range by 0.05.
        """
        self._gait_period_range[0] = max(
            self._gait_period_range[0] - 0.05, GAIT_PERIOD_RANGE[0]
        )
        self._gait_period_range[1] = min(
            self._gait_period_range[1] + 0.05, GAIT_PERIOD_RANGE[1]
        )

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

        # Convert env_ids to tensor if it's a list
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=gs.device, dtype=torch.long)

        if self._num_gaits == 1:
            # Only one gait available - set all to the same gait
            self._set_gait(0, env_ids)
            self._gait_selected[env_ids] = 0
        else:
            # Generate a random list of gait indices
            gait_indices = self._generate_random_gait_indices(len(env_ids))
            for gait_idx in range(self._num_gaits):
                mask = gait_indices == gait_idx
                if mask.any():
                    selected_envs = env_ids[mask]
                    self._set_gait(gait_idx, selected_envs)
                    self._gait_selected[selected_envs] = gait_idx

    def build(self):
        """
        Get foot link indices
        """
        super().build()
        self._robot: RigidEntity = getattr(self.env, self._robot_entity_attr)
        for i, key in enumerate(self._foot_names.keys()):
            foot_link_name = self._foot_names[key]
            link = self._robot.get_link(foot_link_name)
            self._foot_links.insert(i, link)
            self._foot_link_idx.insert(i, link.idx_local)

    def step(self):
        """
        Increment the gait time and phase
        """
        super().step()
        self._log_metrics()

        # Get foot height off the terrain
        foot_pos = self._robot.get_links_pos(links_idx_local=self._foot_link_idx)
        if self._terrain_mgr is not None:
            for i in range(8):
                terrain_height = self._terrain_mgr.get_terrain_height(
                    foot_pos[:, i, 0], foot_pos[:, i, 1]
                )
                foot_pos[:, i, 2] += terrain_height
        self._curr_foot_height = foot_pos[:, :, 2]

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
                self._clock_input,
            ],
            dim=-1,
        )

    def use_gamepad(self, gamepad: Gamepad):
        """
        Control the command using a gamepad.
        Pressing the A button will cycle through the gaits.
        """
        self._gamepad = gamepad
        self._num_gaits = len(GAIT_OFFSETS)
        self._gamepad_select_gait(0)

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
        return torch.exp(-clearance_error / sensitivity)

    def gait_phase_reward(
        self, env: GenesisEnv, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the reward for the feet being in the correct phase.
        """
        reward = 0.0
        for i in range(8):  # for each leg
            reward += self._foot_phase_reward(i, contact_manager).flatten()
        return torch.exp(reward)

    def jump_reward(self, env: GenesisEnv) -> torch.Tensor:
        """
        Calculate the reward for the robot jumping
        Rewards forward velocity + height when all 8 feet are airborne
        """

        # Check if all feet are off the ground (proper jump)
        threshold = 0.05  # 5cm
        num_feet_airborne = (self._curr_foot_height > threshold).sum(dim=-1)
        all_feet_off = (num_feet_airborne == 8).float()
        avg_height = self._curr_foot_height.mean(dim=-1)

        # Forward velocity (x-axis only, not total speed)
        velocity = self._entity_mgr.get_linear_velocity()
        forward_vel = velocity[:, 0]  # x-component only

        # Combine rewards: height + forward velocity, but only when all feet are airborne
        # You can adjust the weighting between height and velocity
        jump_quality = forward_vel + avg_height * 2.0  # weight height 2x

        # Full reward only when properly jumping (all 8 feet off ground)
        # Provide partial reward for learning (gradually increases as more feet lift)
        learning_bonus = (
            num_feet_airborne / 8.0
        ) * 0.3  # 30% partial credit for learning

        return jump_quality * (all_feet_off + learning_bonus)

    """
    Private methods
    """

    def _foot_phase_reward(
        self, foot_idx: int, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the individual foot phase reward
        """
        link = self._foot_links[foot_idx]
        force_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        vel_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )

        # Force / velocity
        force = torch.norm(contact_manager.get_contact_forces(link.idx), dim=-1).view(
            -1, 1
        )
        velocity = torch.norm(link.get_vel(), dim=-1).view(-1, 1)

        # Phase
        phi = (self._gait_phase + self._foot_offset[:, foot_idx].unsqueeze(1)) % 1.0
        phi *= 2 * torch.pi

        swing_indices = (phi >= 0.0) & (phi < torch.pi)
        swing_indices = swing_indices.nonzero().flatten()
        stance_indices = (phi >= torch.pi) & (phi < 2 * torch.pi)
        stance_indices = stance_indices.nonzero().flatten()

        force_weight[swing_indices, :] = -1  # force is penalized during swing phase
        vel_weight[swing_indices, :] = 0  # speed is not penalized during swing phase
        force_weight[stance_indices, :] = 0  # force is not penalized during stance
        vel_weight[stance_indices, :] = -1  # speed is penalized during stance phase

        return vel_weight * velocity + force_weight * force

    def _set_gait(self, gait_idx: int, env_ids: torch.Tensor | None = None):
        """
        Set the gait for a batch of environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)

        gait_name = list(GAIT_OFFSETS.keys())[gait_idx]

        # Set the gait index
        self._gait_idx[env_ids] = gait_idx

        # Define the foot offsets for the selected gait (vectorized assignment)
        gait_offsets = list(GAIT_OFFSETS[gait_name].values())
        self._foot_offset[env_ids] = torch.tensor(
            gait_offsets, device=gs.device, dtype=torch.float
        )

        # Foot clearance
        self._target_foot_height[env_ids, 0] = torch.empty(
            len(env_ids), device=gs.device
        ).uniform_(*self._foot_clearance_range)

        # Gait period
        self._gait_period[env_ids, 0] = torch.empty(
            len(env_ids), device=gs.device
        ).uniform_(*self._gait_period_range)

        # For jumping, update the velocity command manager to only allow positive linear velocity in the x direction
        if gait_name == "jump":
            self._vel_command_mgr.set_command("lin_vel_x", 0.0, env_ids)
            self._vel_command_mgr.set_command("ang_vel_z", 0.0, env_ids)
            lin_vel_x = self._vel_command_mgr.get_command("lin_vel_x")[env_ids]
            self._vel_command_mgr.set_command(
                "lin_vel_x", lin_vel_x.clamp_(min=0.5), env_ids
            )

    def _generate_random_gait_indices(self, num: int) -> torch.Tensor:
        """
        Pick a list of random gait indices, with the most recent gaits having a higher probability of
        being picked.
        If self._all_gaits_learned is True, all gaits are equally weighted.
        """

        if not self._all_gaits_learned:
            # If we haven't learned all gaits yet, weight the recent gaits exponentially higher
            weights = torch.arange(self._num_gaits, device=gs.device).exp()
        else:
            # Otherwise, all gaits are equally weighted
            weights = torch.ones(self._num_gaits, device=gs.device)

        # Normalize: the sum of all weights should be 1
        weights /= weights.sum()
        weights = weights[: self._num_gaits].expand(num, -1)

        return torch.multinomial(weights, 1).squeeze(-1)

    def _process_gamepad_input(self):
        """
        Select a new gait when the A button is pressed.
        """
        if "A" in self._gamepad.state.buttons:
            self._gamepad_btn_pressed = True
        elif self._gamepad_btn_pressed:
            self._gamepad_btn_pressed = False
            gait_idx = (self._gamepad_gait_idx + 1) % self._num_gaits
            self._gamepad_select_gait(gait_idx)

    def _gamepad_select_gait(self, gait_idx: GaitName):
        """
        Select a new gait when the A button is pressed.
        """
        self._gamepad_gait_idx = gait_idx
        gait_name = list(GAIT_OFFSETS.keys())[gait_idx]
        print(f"💃 Selecting gait: {gait_name}")
        env_idx = 0

        # Define the foot offsets for the selected gait (vectorized assignment)
        gait_offsets = torch.tensor(
            list(GAIT_OFFSETS[gait_name].values()), device=gs.device, dtype=torch.float
        )
        self._foot_offset[env_idx] = gait_offsets

        # Foot clearance
        mid_foot_clearance = (FOOT_CLEARANCE_RANGE[0] + FOOT_CLEARANCE_RANGE[1]) / 2
        self._target_foot_height[env_idx, 0] = mid_foot_clearance

        # Gait period
        mid_gait_period = (GAIT_PERIOD_RANGE[0] + GAIT_PERIOD_RANGE[1]) / 2
        self._gait_period[env_idx, 0] = mid_gait_period

    def _log_metrics(self):
        self.env.extras[self.env.extras_logging_key][
            "Metrics / num_gaits"
        ] = self._num_gaits

        # Log gait distribution if multiple gaits are active
        gait_names = list(GAIT_OFFSETS.keys())
        for i, gait_name in enumerate(gait_names):
            count = (self._gait_selected == i).sum()
            self.env.extras[self.env.extras_logging_key][
                f"Metrics / gait_{gait_name}_envs"
            ] = count
