"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class SpiderRobotEnv(MujocoEnv):
    """
    Simplified environment with curriculum learning and cleaner reward structure.

    Curriculum stages:
    1. Balance: Learn to stand and maintain stability
    2. Movement: Learn to move forward while stable
    3. Efficiency: Optimize gait patterns and energy usage
    """

    def __init__(
        self,
        xml_file,
        frame_skip=5,
        render_mode=None,
        curriculum_stage=1,
        **kwargs,
    ):
        self.curriculum_stage = curriculum_stage
        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = 0.0

        # Simplified tracking
        self.previous_action = None
        self.feet_contact_count = 0

        # Core parameters
        self.target_height = 0.134  # Will be set on reset
        self.max_torque = 8.0
        self.position_gain = 15.0
        self.velocity_gain = 0.8

        # Initialize MuJoCo environment
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),  # Simplified observation space
            dtype=np.float64,
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(24,),  # Normalized actions
            dtype=np.float32,
        )

    def step(self, action):
        """Simplified step with curriculum-aware rewards."""
        # Convert normalized actions to joint positions
        joint_ranges = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if self.model.jnt_limited[joint_id]:
                joint_ranges.append(
                    (
                        self.model.jnt_range[joint_id, 0],
                        self.model.jnt_range[joint_id, 1],
                    )
                )
            else:
                joint_ranges.append((-np.pi, np.pi))

        # Scale actions to joint ranges
        target_positions = []
        for i, (low, high) in enumerate(joint_ranges):
            scaled_pos = low + (action[i] + 1.0) * 0.5 * (high - low)
            target_positions.append(scaled_pos)
        target_positions = np.array(target_positions)

        # PD control
        current_positions = self.data.qpos[7:31]
        current_velocities = self.data.qvel[6:30]

        torques = (
            self.position_gain * (target_positions - current_positions)
            - self.velocity_gain * current_velocities
        )
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        self.do_simulation(torques, self.frame_skip)

        self.episode_length += 1
        observation = self._get_obs()
        reward = self._compute_curriculum_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        self.previous_action = action.copy()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Reset with slight randomization."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Small random perturbations
        qpos[7:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq - 7)
        qvel[6:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv - 6)

        self.set_state(qpos, qvel)

        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = self.data.qpos[0]
        self.target_height = self.data.qpos[2]
        self.previous_action = np.zeros(24)
        self.feet_contact_count = 0

        return self._get_obs()

    def _get_obs(self):
        """Simplified observation focusing on essential information."""
        # Core observations
        body_height = self.data.qpos[2]
        body_velocity = self.data.qvel[:3]

        # Orientation as rotation matrix (flattened)
        orientation = self.data.xquat[1]  # Assuming body ID 1

        # Joint positions and velocities (normalized)
        joint_positions = self.data.qpos[7:31]
        joint_velocities = self.data.qvel[6:30]

        # Foot contacts (binary)
        foot_contacts = self._get_foot_contacts()

        obs = np.concatenate(
            [
                [body_height],  # 1
                body_velocity,  # 3
                orientation,  # 4
                joint_positions,  # 24
                joint_velocities / 10.0,  # 24 (normalized)
                foot_contacts,  # 8
            ]
        )

        return obs[:48]  # Ensure correct size

    def _compute_curriculum_reward(self, action):
        """Compute reward based on curriculum stage."""
        # Common measurements
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)

        forward_velocity = self.data.qvel[0]
        lateral_velocity = abs(self.data.qvel[1])

        orientation = self.data.xquat[1]
        upright_error = 1.0 - orientation[3]  # Assuming w component should be 1

        # Stage 1: Balance (focus on stability)
        if self.curriculum_stage == 1:
            height_reward = 10.0 * np.exp(-50 * height_error**2)
            upright_reward = 5.0 * np.exp(-10 * upright_error**2)
            stillness_reward = 2.0 * np.exp(-np.sum(self.data.qvel**2))

            reward = height_reward + upright_reward + stillness_reward + 1.0

        # Stage 2: Movement (add forward progress)
        elif self.curriculum_stage == 2:
            height_reward = 5.0 * np.exp(-30 * height_error**2)
            upright_reward = 3.0 * np.exp(-10 * upright_error**2)
            forward_reward = 3.0 * np.clip(forward_velocity, -0.5, 2.0)
            lateral_penalty = -2.0 * lateral_velocity

            reward = (
                height_reward + upright_reward + forward_reward + lateral_penalty + 0.5
            )

        # Stage 3: Efficiency (optimize everything)
        else:
            height_reward = 3.0 * np.exp(-20 * height_error**2)
            upright_reward = 2.0 * np.exp(-10 * upright_error**2)
            forward_reward = 5.0 * np.clip(forward_velocity, -0.5, 3.0)

            # Energy efficiency
            if self.previous_action is not None:
                action_change = np.sum((action - self.previous_action) ** 2)
                smoothness_reward = 1.0 * np.exp(-0.5 * action_change)
            else:
                smoothness_reward = 0.0

            # Gait quality (simple version)
            foot_contacts = self._get_foot_contacts()
            num_contacts = np.sum(foot_contacts)
            if 3 <= num_contacts <= 5:
                gait_reward = 2.0
            else:
                gait_reward = 0.0

            reward = (
                height_reward
                + upright_reward
                + forward_reward
                + smoothness_reward
                + gait_reward
            )

        return reward

    def _get_foot_contacts(self):
        """Get binary foot contact information."""
        contacts = np.zeros(8, dtype=np.float32)

        # Check each contact in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Simplified: assume geom IDs 10-17 are feet
            for foot_idx in range(8):
                foot_geom_id = 10 + foot_idx
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    contacts[foot_idx] = 1.0

        return contacts

    def _is_terminated(self):
        """Simple termination conditions."""
        height = self.data.qpos[2]
        if height < 0.05:
            return True

        # Check for simulation instability
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True

        return False

    def _get_info(self):
        """Return training information."""
        return {
            "episode_length": self.episode_length,
            "height": self.data.qpos[2],
            "forward_velocity": self.data.qvel[0],
            "distance_traveled": self.data.qpos[0] - self.initial_x_position,
            "curriculum_stage": self.curriculum_stage,
        }

    def set_curriculum_stage(self, stage):
        """Update curriculum stage during training."""
        self.curriculum_stage = max(1, min(3, stage))
