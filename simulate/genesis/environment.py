"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import torch
import numpy as np
from gymnasium import spaces
from typing import Sequence, Any

import genesis as gs

from robo_genesis import (
    GenesisEnv,
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
)
from robo_genesis.utils import robot_projected_gravity, robot_ang_vel, robot_lin_vel
from robo_genesis.mdp import rewards, terminations


TARGET_HEIGHT = 0.15
PD_KP = 50.0
PD_KV = 0.5
MAX_TORQUE = 8.0

INITIAL_BODY_POSITION = [0.0, 0.0, 0.135]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

INIT_FEMUR_POS = 0.5
INIT_TIBIA_POS = 0.6
INIT_JOINT_POS = {
    "Leg1_Hip": -1.0,
    "Leg1_Femur": INIT_FEMUR_POS,
    "Leg1_Tibia": INIT_TIBIA_POS,
    "Leg2_Hip": -1.0,
    "Leg2_Femur": INIT_FEMUR_POS,
    "Leg2_Tibia": INIT_TIBIA_POS,
    "Leg3_Hip": 1.0,
    "Leg3_Femur": INIT_FEMUR_POS,
    "Leg3_Tibia": INIT_TIBIA_POS,
    "Leg4_Hip": 1.0,
    "Leg4_Femur": INIT_FEMUR_POS,
    "Leg4_Tibia": INIT_TIBIA_POS,
    "Leg5_Hip": 1.0,
    "Leg5_Femur": INIT_FEMUR_POS,
    "Leg5_Tibia": INIT_TIBIA_POS,
    "Leg6_Hip": 1.0,
    "Leg6_Femur": INIT_FEMUR_POS,
    "Leg6_Tibia": INIT_TIBIA_POS,
    "Leg7_Hip": -1.0,
    "Leg7_Femur": INIT_FEMUR_POS,
    "Leg7_Tibia": INIT_TIBIA_POS,
    "Leg8_Hip": -1.0,
    "Leg8_Femur": INIT_FEMUR_POS,
    "Leg8_Tibia": INIT_TIBIA_POS,
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


class SpiderRobotEnv(GenesisEnv):
    """
    SpiderBot environment for Genesis
    """

    num_actions = 24

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int = 12,
        headless: bool = True,
    ):
        super().__init__(num_envs, dt, max_episode_length_s, headless)
        self.default_dof_pos = torch.tensor(
            list(INIT_JOINT_POS.values()), device=gs.device
        )

        # Command manager: instruct the robot to move in a certain direction
        self.command_manager = VelocityCommandManager(
            self,
            lin_vel_x_range=[-1.0, 1.0],
            lin_vel_y_range=[-1.0, 1.0],
            ang_vel_z_range=[0.5, 0.5],
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        # Rewards
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            reward_cfg={
                "Linear Z velocity": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z,
                },
                "Base height": {
                    "weight": -100.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": TARGET_HEIGHT,
                    },
                },
                "Action rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate,
                },
                "Similar to default": {
                    "weight": -0.1,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "dof_idx": self._get_dof_idx,
                        "default_dof_pos": self.default_dof_pos,
                    },
                },
                "Cmd linear velocity": {
                    "weight": 2.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.command_manager,
                    },
                },
                "Cmd angular velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.command_manager,
                    },
                },
            },
        )

        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "Timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "Bad angle": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 0.7,
                    },
                },
            },
        )

    @property
    def observation_space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(84,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    def construct_scene(self) -> gs.Scene:
        """Add the robot to the scene."""
        scene = super().construct_scene()

        # add robot
        self.robot = scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        return scene

    def build_scene(self) -> None:
        """
        Builds the scene after all entities have been added (via construct_scene).
        This operation is required before running the simulation.
        """
        super().build_scene()

        # Actuator indices
        self.dof_idx = [
            self.robot.get_joint(name).dof_start for name in INIT_JOINT_POS.keys()
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

        # Cache position tensors
        self.base_init_pos = torch.tensor(INITIAL_BODY_POSITION, device=gs.device)
        self.base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device).reshape(
            1, -1
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        super().step(actions)

        # Validate actions
        if torch.isnan(actions).any():
            print(f"ERROR: NaN actions received in step function! Actions: {actions}")
        if torch.isinf(actions).any():
            print(
                f"ERROR: Infinite actions received in step function! Actions: {actions}"
            )

        # Execute simulation step
        self._control_dof_position_from_actions(actions)
        self.scene.step()

        # Termination, rewards
        terminated, truncated, reset_env_idx = self.termination_manager.step()
        rewards = self.reward_manager.step()

        # Command manager
        self.command_manager.step()

        # Finish up
        self.reset(reset_env_idx)
        self._get_observations()
        return self.obs_buf, rewards, terminated, truncated, {}

    def reset(
        self,
        envs_idx: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or more environments.
        """
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        if envs_idx.numel() > 0:
            # Managers
            self.command_manager.reset(envs_idx)
            self.termination_manager.reset(envs_idx)
            self.reward_manager.reset(envs_idx)

            # Reset robot
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_quat[envs_idx] = self.base_init_quat
            self.dof_pos[envs_idx] = self.default_dof_pos
            self.robot.zero_all_dofs_velocity(envs_idx)
            self.robot.set_pos(self.base_pos[envs_idx], envs_idx=envs_idx)
            self.robot.set_quat(self.base_quat[envs_idx], envs_idx=envs_idx)
            self.robot.set_dofs_position(
                position=self.dof_pos[envs_idx],
                dofs_idx_local=self.dof_idx,
                envs_idx=envs_idx,
            )

        obs = self._get_observations()
        return obs, {}

    def render(self):
        pass

    def close(self):
        """Close the environment."""
        self.scene.reset()

    def _get_observations(self):
        """Environment observations"""
        self.obs_buf = torch.cat(
            [
                self.command_manager.command,  # 3
                robot_ang_vel(self),  # 3
                robot_lin_vel(self),  # 3
                robot_projected_gravity(self),  # 3
                self.robot.get_dofs_position(self.dof_idx),  # 24
                self.robot.get_dofs_velocity(self.dof_idx),  # 24
                self.actions,  # 24
            ],
            dim=-1,
        )
        return self.obs_buf

    def _control_dof_position_from_actions(self, actions: torch.Tensor):
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

    def _get_dof_idx(self) -> Sequence[int]:
        """
        Helper function used by the reward manager to get the DOF indices.
        This is necessary since the rewards are defined before the scene is built and the DOF indices are known.
        """
        return self.dof_idx
