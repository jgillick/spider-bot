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

from genesis_forge import GenesisEnv
from genesis_forge.managers import (
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
    DofPositionActionManager,
)
from genesis_forge.utils import robot_projected_gravity, robot_ang_vel, robot_lin_vel
from genesis_forge.mdp import rewards, terminations


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

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int = 12,
        headless: bool = True,
    ):
        super().__init__(num_envs, dt, max_episode_length_s, headless)

        # Define the DOF actuators
        self.action_manager = DofPositionActionManager(
            self,
            joint_names=".*",
            default_pos={
                # Hip joints
                "Leg[1-2]_Hip": -1.0,
                "Leg[3-6]_Hip": 1.0,
                "Leg[7-8]_Hip": -1.0,
                # Femur joints
                "Leg[1-8]_Femur": 0.5,
                # Tibia joints
                "Leg[1-8]_Tibia": 0.6,
            },
            pd_kp={".*": 50},
            pd_kv={".*": 0.5},
            max_force={".*": 8.0},
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
                "Similar to default": {
                    "weight": -0.1,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "dof_action_manager": self.action_manager,
                    },
                },
                "Action rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate,
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

        # Cache position buffers
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
        return self.action_manager.action_space

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

    def build(self) -> None:
        """
        Builds the scene after all entities have been added (via construct_scene).
        This operation is required before running the simulation.
        """
        super().build()
        self.action_manager.build()

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        super().step(actions)

        # Execute the actions and a simulation step
        self.action_manager.step(actions)
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
            self.action_manager.reset(envs_idx)

            # Reset robot
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_quat[envs_idx] = self.base_init_quat
            self.robot.zero_all_dofs_velocity(envs_idx)
            self.robot.set_pos(self.base_pos[envs_idx], envs_idx=envs_idx)
            self.robot.set_quat(self.base_quat[envs_idx], envs_idx=envs_idx)

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
                self.action_manager.get_dofs_position(),  # 24
                self.action_manager.get_dofs_velocity(),  # 24
                self.actions,  # 24
            ],
            dim=-1,
        )
        return self.obs_buf
