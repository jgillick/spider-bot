"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import math
import torch
import numpy as np
from gymnasium import spaces
from typing import Sequence, Any

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

from robo_genesis import GenesisEnv


TARGET_HEIGHT = 0.15
REWARDS = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
}

INITIAL_BODY_POSITION = [0.0, 0.0, 0.145]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
FEMUR_INIT_POSITION = 0.5
TIBIA_INIT_POSITION = 0.6
INITIAL_JOINT_POSITIONS = {
    "Leg1_Hip": -1.0,
    "Leg1_Femur": FEMUR_INIT_POSITION,
    "Leg1_Tibia": TIBIA_INIT_POSITION,
    "Leg2_Hip": -1.0,
    "Leg2_Femur": FEMUR_INIT_POSITION,
    "Leg2_Tibia": TIBIA_INIT_POSITION,
    "Leg3_Hip": 1.0,
    "Leg3_Femur": FEMUR_INIT_POSITION,
    "Leg3_Tibia": TIBIA_INIT_POSITION,
    "Leg4_Hip": 1.0,
    "Leg4_Femur": FEMUR_INIT_POSITION,
    "Leg4_Tibia": TIBIA_INIT_POSITION,
    "Leg5_Hip": 1.0,
    "Leg5_Femur": FEMUR_INIT_POSITION,
    "Leg5_Tibia": TIBIA_INIT_POSITION,
    "Leg6_Hip": 1.0,
    "Leg6_Femur": FEMUR_INIT_POSITION,
    "Leg6_Tibia": TIBIA_INIT_POSITION,
    "Leg7_Hip": -1.0,
    "Leg7_Femur": FEMUR_INIT_POSITION,
    "Leg7_Tibia": TIBIA_INIT_POSITION,
    "Leg8_Hip": -1.0,
    "Leg8_Femur": FEMUR_INIT_POSITION,
    "Leg8_Tibia": TIBIA_INIT_POSITION,
}

PD_KP = 50.0
PD_KV = 0.5
MAX_TORQUE = 8.0

COMMANDS = {
    "lin_vel_x_range": [0.5, 0.5],
    "lin_vel_y_range": [0.5, 0.5],
    "ang_vel_range": [0.5, 0.5],
}
COMMAND_REWARD_SIGMA = 0.25

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class SpiderRobotEnv(GenesisEnv):
    """
    SpiderBot environment for Genesis
    """

    base_init_pos: torch.Tensor
    base_init_quat: torch.Tensor

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int = 15,
        headless: bool = True,
    ):
        super().__init__(num_envs, dt, max_episode_length_s, headless)

        # Observation/Action spaces
        self.num_actions = 24
        self.num_observations = 81
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )

        # Prepare reward functions
        self.reward_func, self.reward_weight, self._episode_rewards = (
            dict(),
            dict(),
            dict(),
        )
        for name, base_weight in REWARDS.items():
            if base_weight == 0.0:
                continue
            self.reward_weight[name] = base_weight * self.dt
            self.reward_func[name] = getattr(self, "_reward_" + name)
            self._episode_rewards[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # Initialize the scene
        self._init_buffers()

    def construct_scene(self) -> gs.Scene:
        """Add the robot to the scene."""
        scene = super().construct_scene()

        # add robot
        self.base_init_pos = torch.tensor(INITIAL_BODY_POSITION, device=gs.device)
        self.base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=self.base_init_pos,
                quat=self.base_init_quat,
            ),
        )

        return scene

    def build_scene(self) -> None:
        """Builds the scene once all entities have been added (via construct_scene). This operation is required before running the simulation."""
        super().build_scene()

        # Actuator indices
        self.dof_idx = [
            self.robot.get_joint(name).dof_start
            for name in INITIAL_JOINT_POSITIONS.keys()
        ]

        # Set gains and torque limit
        self.robot.set_dofs_kp([PD_KP] * self.num_actions, self.dof_idx)
        self.robot.set_dofs_kv([PD_KV] * self.num_actions, self.dof_idx)
        self.robot.set_dofs_force_range(
            [-MAX_TORQUE] * self.num_actions,
            [MAX_TORQUE] * self.num_actions,
            self.dof_idx,
        )

        # Get position Limits and convert to shape (num_envs, limit)
        self.actuator_limits_lower, self.actuator_limits_upper = (
            self.robot.get_dofs_limit(self.dof_idx)
        )
        self.actuator_limits_lower = self.actuator_limits_lower.unsqueeze(0).expand(
            self.num_envs, -1
        )
        self.actuator_limits_upper = self.actuator_limits_upper.unsqueeze(0).expand(
            self.num_envs, -1
        )

        # Initial positions
        self.default_dof_pos = torch.tensor(
            list(INITIAL_JOINT_POSITIONS.values()),
            device=gs.device,
            dtype=gs.tc_float,
        )

    def _init_buffers(self):
        """Initialize buffers that will track the state of the environments."""
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)

        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )

        self.commands = torch.zeros(
            (self.num_envs, len(COMMANDS)), device=gs.device, dtype=gs.tc_float
        )

    def step(self, actions: torch.Tensor):
        """Perform a step in the environment."""
        super().step(actions)

        # Validate actions
        if torch.isnan(actions).any():
            print(f"ERROR: NaN actions received in step function! Actions: {actions}")
        if torch.isinf(actions).any():
            print(
                f"ERROR: Infinite actions received in step function! Actions: {actions}"
            )

        # Execute simulation step
        self._set_actuator_positions(actions)
        self.scene.step()

        # Update state buffers
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.dof_idx)

        # Termination, rewards, and observations
        terminations, timeouts = self._check_terminated()
        rewards, rewards_logs = self._calculate_rewards()

        # Reset environments that terminated or timed out
        resets = timeouts | terminations
        reset_idx = resets.nonzero(as_tuple=False).reshape((-1,))
        self.reset(reset_idx)

        self.get_observations()
        info = {
            "logs": {
                "episode": rewards_logs,
            }
        }
        return self.obs_buf, rewards, terminations, timeouts, info

    def reset(
        self,
        env_ids: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset one or more environments."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=gs.device)

        # reset base
        self.base_pos[env_ids] = self.base_init_pos
        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)
        self.robot.zero_all_dofs_velocity(env_ids)
        self.robot.set_pos(self.base_pos[env_ids], zero_velocity=True, envs_idx=env_ids)
        self.robot.set_quat(
            self.base_quat[env_ids], zero_velocity=True, envs_idx=env_ids
        )

        # reset dofs
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.dof_idx,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        self.base_lin_vel[env_ids] = 0
        self.base_ang_vel[env_ids] = 0

        # Set new command
        self._resample_commands(env_ids)

        obs = self.get_observations()
        return obs, {}

    def render(self):
        pass

    def close(self):
        """Close the environment."""
        self.scene.reset()

    def _set_actuator_positions(self, actions: torch.Tensor):
        """Convert actions to position commands, and send them to the actuators."""
        self.actions = actions.clamp(-1.0, 1.0)
        lower = self.actuator_limits_lower
        upper = self.actuator_limits_upper

        # Get the center of each actuator range
        center = (upper + lower) * 0.5

        # Convert the action to absolute position (from -1 - 1, to position)
        self.target_positions = (
            self.actions
            * (self.actuator_limits_upper - self.actuator_limits_lower)
            * 0.5
            + center
        )

        # Set target positions
        self.robot.control_dofs_position(self.target_positions, self.dof_idx)

    def get_observations(self):
        """Environment observations"""
        self.obs_buf = torch.cat(
            [
                self.commands,  # 3
                self.base_ang_vel,  # 3
                self.projected_gravity,  # 3
                self.dof_pos,  # 24
                self.dof_vel,  # 24
                self.actions,  # 24
            ],
            dim=-1,
        )
        return self.obs_buf

    def _check_terminated(self):
        """Check if episode should terminate (robot has fallen or become unstable)."""

        # -- Timeout
        timeouts = self.episode_length > self.max_episode_length

        # -- Termination
        # Check if robot has flipped over
        terminations = torch.abs(self.base_euler[:, 1]) > 90  # Degrees
        terminations |= torch.abs(self.base_euler[:, 0]) > 90  # Degrees

        return terminations, timeouts

    def _calculate_rewards(self):
        """Compute the total rewards"""

        self.rew_buf[:] = 0.0
        logs = dict()
        for name, reward_func in self.reward_func.items():
            rew = reward_func() * self.reward_weight[name]
            self.rew_buf += rew
            self._episode_rewards[name] += rew
            logs[f"Rewards/{name}"] = self._episode_rewards[name]

        return self.rew_buf, logs

    def _track_episode_rewards(self, env_idx: Sequence[int]):
        """Compute the total rewards"""
        for key in REWARDS.keys():
            # Get episode lengths and ensure they're valid
            episode_lengths = self.episode_length[env_idx]

            # Handle edge cases where episode lengths might be zero
            valid_mask = episode_lengths > 0
            if torch.any(valid_mask):
                # Calculate average for each episode based on its actual length
                episode_avg = torch.zeros_like(self._episode_rewards[key][env_idx])
                episode_avg[valid_mask] = (
                    self._episode_rewards[key][env_idx][valid_mask]
                    / episode_lengths[valid_mask]
                )
                # Take the mean across valid episodes only
                episode_mean = torch.mean(episode_avg[valid_mask])
                self.track_data(f"Episode_Reward/{key}", episode_mean)

            # reset episodic sum
            self._episode_rewards[key][env_idx] = 0.0

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(
            *COMMANDS["lin_vel_x_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *COMMANDS["lin_vel_y_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *COMMANDS["ang_vel_range"], (len(envs_idx),), gs.device
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / COMMAND_REWARD_SIGMA)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / COMMAND_REWARD_SIGMA)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - TARGET_HEIGHT)
