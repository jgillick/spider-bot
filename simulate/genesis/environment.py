"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import math
import torch
import numpy as np
from gymnasium import spaces
from typing import Callable, Sequence

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


TARGET_HEIGHT = 0.14
REWARDS = {
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
}

FEMUR_INIT_POSITION = 0.5
TIBIA_INIT_POSITION = -0.5
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


INITIAL_POSITION = [0.0, 0.0, 0.2]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

PD_KP = 50.0
PD_KV = 0.5
MAX_TORQUE = 8.0

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


class SpiderRobotEnv:
    """
    Gymnasium compatible environment for SpiderBot.
    """

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_length_s: int = 16,
    ):
        self.device = gs.device
        self.num_envs = num_envs
        self.dt = 1 / 100  # Hz
        self.max_episode_length = math.ceil(max_episode_length_s / self.dt)
        self._track_data: Callable[[str, float], None] = None

        # Environment spaces
        self.num_obs = 78
        self.num_actions = 24
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Prepare reward functions
        self.reward_func, self.reward_weight, self._episode_rewards = (
            dict(),
            dict(),
            dict(),
        )
        for name, base_weight in REWARDS.items():
            self.reward_weight[name] = base_weight * self.dt
            self.reward_func[name] = getattr(self, "_reward_" + name)
            self._episode_rewards[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # Initialize the scene
        self._init_scene()
        self._init_actuators()
        self._init_buffers()

    def _init_scene(self):
        """Create the scene."""
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=True,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(INITIAL_POSITION, device=gs.device)
        self.base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=self.num_envs)

    def _init_actuators(self):
        """Fetch the actuators and set the Kp/Kv values."""
        # Actuator indices
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_start
            for name in INITIAL_JOINT_POSITIONS.keys()
        ]

        # Gains and torque limits
        self.robot.set_dofs_kp([PD_KP] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([PD_KV] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_force_range(
            [-MAX_TORQUE] * self.num_actions,
            [MAX_TORQUE] * self.num_actions,
            self.motors_dof_idx,
        )

        # Position Limits
        self.actuator_limits_lower, self.actuator_limits_upper = (
            self.robot.get_dofs_limit(self.motors_dof_idx)
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
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def track_data(self, tag: str, value: float):
        """Log data to tensorboard."""
        if self._track_data is not None:
            self._track_data(tag, value)

    def set_data_tracker(self, track_data_fn: Callable[[str, float], None]):
        """Set the function which logs data to tensorboard."""
        self._track_data = track_data_fn

    def step(self, actions: torch.Tensor):
        """Perform a step in the environment."""

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
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        # Fix the quaternion transformation to ensure proper shapes
        inv_base_init_quat_expanded = self.inv_base_init_quat.unsqueeze(0).expand(
            self.num_envs, -1
        )
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                inv_base_init_quat_expanded,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Termination, rewards, and observations
        terminations, timeouts = self._check_terminated()
        self._calculate_rewards()
        self._get_observation()

        # Retain previous state
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, terminations, timeouts, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self._reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    def render(self):
        pass

    def close(self):
        """Close the environment."""
        pass

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
        self.robot.control_dofs_position(self.target_positions, self.motors_dof_idx)

    def _get_observation(self):
        """Environment observations"""
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel,  # 3
                self.projected_gravity,  # 3
                self.dof_pos,  # 24
                self.dof_vel,  # 24
                self.actions,  # 24
            ],
            dim=-1,
        )
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf

    def _check_terminated(self):
        """Check if episode should terminate (robot has fallen or become unstable)."""

        # -- Timeout
        timeouts = self.episode_length_buf > self.max_episode_length
        time_out_idx = timeouts.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(
            timeouts, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        # -- Termination
        # Check if robot has flipped over
        terminations = torch.abs(self.base_euler[:, 1]) > 90  # Degrees
        terminations |= torch.abs(self.base_euler[:, 0]) > 90  # Degrees

        term_idx = terminations.nonzero(as_tuple=False).reshape((-1,))
        self.extras["terminations"] = torch.zeros_like(
            terminations, device=gs.device, dtype=gs.tc_float
        )
        self.extras["terminations"][term_idx] = 1.0

        # Reset if necessary
        self.reset_buf[:] = timeouts | terminations
        self._reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        return terminations, timeouts

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Track episode rewards
        self._track_episode_rewards(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.zero_all_dofs_velocity(envs_idx)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0

    def _calculate_rewards(self):
        """Compute the total rewards"""
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_func.items():
            rew = reward_func() * self.reward_weight[name]
            self.rew_buf += rew
            self._episode_rewards[name] += rew

    def _track_episode_rewards(self, env_idx: Sequence[int]):
        """Compute the total rewards"""
        for key in REWARDS.keys():
            # Get episode lengths and ensure they're valid
            episode_lengths = self.episode_length_buf[env_idx]

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
