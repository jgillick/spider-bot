"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from mujoco import mjtObj, mj_name2id

INITIAL_JOINT_POSITIONS = (
    "-1",  # Leg 1 - Hip
    "0.75",  # Leg 1 - Femur
    "1.0",  # Leg 1 - Tibia
    "-1",  # Leg 2 - Hip
    "0.75",  # Leg 2 - Femur
    "1.0",  # Leg 2 - Tibia
    "1",  # Leg 3 - Hip
    "0.75",  # Leg 3 - Femur
    "1.0",  # Leg 3 - Tibia
    "1",  # Leg 4 - Hip
    "0.75",  # Leg 4 - Femur
    "1.0",  # Leg 4 - Tibia
    "1",  # Leg 5 - Hip
    "0.75",  # Leg 5 - Femur
    "1.0",  # Leg 5 - Tibia
    "1",  # Leg 6 - Hip
    "0.75",  # Leg 6 - Femur
    "1.0",  # Leg 6 - Tibia
    "-1.0",  # Leg 7 - Hip
    "0.75",  # Leg 7 - Femur
    "1.0",  # Leg 7 - Tibia
    "-1.0",  # Leg 8 - Hip
    "0.75",  # Leg 8 - Femur
    "1.0",  # Leg 8 - Tibia
)


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

        # Episode stability tracking
        self.stable_steps = 0
        self.total_steps = 0

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

        # Cache foot geom IDs by name
        self._cache_foot_geom_ids()

        # Set initial joint positions
        self._set_initial_joint_positions()

    def _set_initial_joint_positions(self):
        """Set the initial joint positions from the constant."""
        qpos = self.data.qpos.copy()

        # Set joint positions from the constant
        for i, pos_str in enumerate(INITIAL_JOINT_POSITIONS):
            if i < len(qpos) - 7:  # Ensure we don't go out of bounds
                qpos[7 + i] = float(pos_str)

        # Update the state
        self.set_state(qpos, self.data.qvel)

        # Debug: Print initial joint positions
        print(
            f"🦵 Set initial joint positions: {qpos[7:7+len(INITIAL_JOINT_POSITIONS)]}"
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
        self.total_steps += 1

        # Track stability
        if self._is_stable():
            self.stable_steps += 1

        observation = self._get_obs()
        reward = self._compute_curriculum_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        self.previous_action = action.copy()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Reset with predefined joint positions and slight randomization."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Set initial joint positions from the constant
        for i, pos_str in enumerate(INITIAL_JOINT_POSITIONS):
            if i < len(qpos) - 7:  # Ensure we don't go out of bounds
                qpos[7 + i] = float(pos_str)

        # Small random perturbations to joint positions only
        joint_perturbations = self.np_random.uniform(
            low=-0.005, high=0.005, size=len(INITIAL_JOINT_POSITIONS)
        )
        for i in range(min(len(joint_perturbations), len(qpos) - 7)):
            qpos[7 + i] += joint_perturbations[i]

        # Small random perturbations to velocities
        qvel[6:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv - 6)

        self.set_state(qpos, qvel)

        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = self.data.qpos[0]
        self.target_height = self.data.qpos[2]
        self.previous_action = np.zeros(24)
        self.feet_contact_count = 0

        # Reset stability tracking
        self.stable_steps = 0
        self.total_steps = 0

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

        # Gyro sensor measurements for stability
        gyro_stability = self._get_gyro_stability()

        # Contact penalties for non-foot parts
        contact_penalty = self._get_contact_penalty()

        # Stage 1: Balance (focus on stability)
        if self.curriculum_stage == 1:
            height_reward = 8.0 * np.exp(
                -100 * height_error**2
            )  # Stronger height control
            upright_reward = 5.0 * np.exp(
                -20 * upright_error**2
            )  # Stronger upright bonus

            # Stronger penalty for joint movement (encourage stillness)
            joint_velocities = self.data.qvel[6:30]
            joint_movement_penalty = -0.5 * np.sum(np.abs(joint_velocities))

            # Stronger reward for minimal body movement
            body_stillness = 5.0 * np.exp(-10 * np.sum(self.data.qvel[:6] ** 2))

            gyro_reward = 6.0 * gyro_stability  # Stronger gyro reward

            # Bonus for maintaining target height
            height_bonus = 2.0 if height_error < 0.005 else 0.0

            # Penalty for action changes (encourage smooth, minimal actions)
            if self.previous_action is not None:
                action_change = np.sum((action - self.previous_action) ** 2)
                action_smoothness_penalty = -0.3 * action_change
            else:
                action_smoothness_penalty = 0.0

            # Calculate stability ratio for reward scaling
            stability_ratio = self.stable_steps / max(1, self.total_steps)

            # Scale rewards based on stability (more stable = higher rewards)
            stability_multiplier = 0.5 + 0.5 * stability_ratio

            reward = (
                height_reward
                + upright_reward
                + body_stillness
                + joint_movement_penalty
                + gyro_reward
                + height_bonus
                + action_smoothness_penalty
                + contact_penalty * 0.2
                + 2.0  # Base reward
            ) * stability_multiplier

        # Stage 2: Movement (add forward progress)
        elif self.curriculum_stage == 2:
            height_reward = 3.0 * np.exp(-30 * height_error**2)
            upright_reward = 2.0 * np.exp(-8 * upright_error**2)

            # Progressive forward reward (encourage consistent forward motion)
            forward_reward = 10.0 * np.clip(
                forward_velocity, 0, 0.5
            )  # More reasonable forward reward

            # Penalize sideways drift
            lateral_penalty = -5.0 * lateral_velocity

            # Gait coordination reward
            foot_contacts = self._get_foot_contacts()
            alternating_contacts = self._get_gait_coordination_reward(foot_contacts)

            gyro_reward = 2.0 * gyro_stability  # Still important for stable walking

            reward = (
                height_reward
                + upright_reward
                + forward_reward
                + lateral_penalty
                + alternating_contacts
                + gyro_reward
                + contact_penalty * 0.2  # Reduce impact
                + 1.5  # Reasonable base reward
            )

        # Stage 3: Efficiency (optimize everything)
        else:
            height_reward = 2.0 * np.exp(-25 * height_error**2)
            upright_reward = 1.5 * np.exp(-8 * upright_error**2)

            # Reward faster forward motion
            forward_reward = 5.0 * np.clip(forward_velocity, 0, 1.0)

            gyro_reward = 1.5 * gyro_stability

            # Energy efficiency (smoother actions)
            if self.previous_action is not None:
                action_change = np.sum((action - self.previous_action) ** 2)
                smoothness_reward = 1.0 * np.exp(-0.5 * action_change)
            else:
                smoothness_reward = 0.0

            # Advanced gait quality
            foot_contacts = self._get_foot_contacts()
            gait_reward = self._get_gait_coordination_reward(foot_contacts) * 0.8

            # Speed consistency bonus
            if hasattr(self, "previous_forward_velocity"):
                speed_consistency = 1.0 * np.exp(
                    -3 * (forward_velocity - self.previous_forward_velocity) ** 2
                )
            else:
                speed_consistency = 0.0
            self.previous_forward_velocity = forward_velocity

            reward = (
                height_reward
                + upright_reward
                + forward_reward
                + gyro_reward
                + smoothness_reward
                + gait_reward
                + speed_consistency
                + contact_penalty * 0.1  # Further reduce impact
                + 1.0  # Reasonable base reward
            )

        return reward

    def _cache_foot_geom_ids(self):
        """Cache foot geom IDs by looking up their names."""
        self.foot_geom_ids = set()

        # Look for foot geoms by name pattern "LegX_Tibia_foot"
        for leg_num in range(1, 9):  # Legs 1-8
            foot_name = f"Leg{leg_num}_Tibia_foot"
            foot_id = mj_name2id(self.model, mjtObj.mjOBJ_GEOM, foot_name)
            if foot_id is not None:
                self.foot_geom_ids.add(foot_id)
            else:
                print(f"⚠️ Warning: Could not find geom '{foot_name}'")

    def _get_contact_penalty(self):
        """Penalize contact with non-foot parts of the robot."""
        penalty = 0.0

        # Check each contact in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Use cached foot geom IDs
            geom1_is_foot = contact.geom1 in self.foot_geom_ids
            geom2_is_foot = contact.geom2 in self.foot_geom_ids

            # Only penalize if BOTH geoms are non-foot parts
            # This allows: foot-foot, foot-ground, foot-robot_part
            # But penalizes: robot_part-robot_part, robot_part-ground
            if not geom1_is_foot and not geom2_is_foot:
                # Check if one of them is ground (geom 0)
                if contact.geom1 == 0 or contact.geom2 == 0:
                    # Moderate penalty for non-foot touching ground
                    penalty -= 2.0
                else:
                    # Very mild penalty for robot parts touching each other
                    penalty -= 0.1

        return penalty

    def _get_gyro_stability(self):
        """Get gyro sensor stability measurement."""
        # Get gyro sensor data (angular velocity)
        gyro_sensor_id = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "gyro_sensor")
        if gyro_sensor_id is not None:
            gyro_data = self.data.sensordata[gyro_sensor_id]

            # Calculate stability based on angular velocity magnitude
            # Lower angular velocity = more stable
            angular_velocity_magnitude = np.linalg.norm(gyro_data)

            # Convert to stability score (0 = unstable, 1 = very stable)
            # Use exponential decay: exp(-angular_velocity)
            stability = np.exp(-2.0 * angular_velocity_magnitude)

            return stability
        else:
            # Fallback if gyro sensor not found
            print("⚠️ Warning: gyro_sensor not found, using fallback stability")
            # Use body angular velocity as fallback
            body_angular_vel = self.data.qvel[3:6]  # Roll, pitch, yaw velocities
            angular_velocity_magnitude = np.linalg.norm(body_angular_vel)
            stability = np.exp(-2.0 * angular_velocity_magnitude)
            return stability

    def _get_gait_coordination_reward(self, foot_contacts):
        """Reward coordinated gait patterns."""
        # Count contacts on each side
        left_contacts = foot_contacts[0:4].sum()  # Legs 1-4
        right_contacts = foot_contacts[4:8].sum()  # Legs 5-8

        # Ideal is alternating contact (some on each side)
        if left_contacts > 0 and right_contacts > 0:
            # Good - both sides have contact
            balance_reward = 0.5

            # Extra reward for diagonal pairs (more stable)
            diagonal_pairs = [
                (foot_contacts[0] and foot_contacts[6]),  # Leg1 & Leg7
                (foot_contacts[1] and foot_contacts[7]),  # Leg2 & Leg8
                (foot_contacts[2] and foot_contacts[4]),  # Leg3 & Leg5
                (foot_contacts[3] and foot_contacts[5]),  # Leg4 & Leg6
            ]
            diagonal_reward = sum(diagonal_pairs) * 0.2

            return balance_reward + diagonal_reward
        else:
            # Poor - all contacts on one side
            return 0.0

    def _get_foot_contacts(self):
        """Get binary foot contact information."""
        contacts = np.zeros(8, dtype=np.float32)

        # Check each contact in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Use cached foot geom IDs
            if (
                contact.geom1 in self.foot_geom_ids
                or contact.geom2 in self.foot_geom_ids
            ):
                # Find which foot this is (assuming they're in order)
                foot_geom_id = (
                    contact.geom1
                    if contact.geom1 in self.foot_geom_ids
                    else contact.geom2
                )
                # Map geom ID to foot index (assuming they're sequential)
                foot_idx = min(foot_geom_id - min(self.foot_geom_ids), 7)
                contacts[foot_idx] = 1.0

        return contacts

    def _is_terminated(self):
        """Check if episode should terminate (robot has fallen or become unstable)."""

        # Terminate if body is too low
        height = self.data.qpos[2]
        if height < 0.05:
            return True

        # Terminate if robot has fallen (body too tilted)
        orientation = self.data.xquat[1]
        upright_error = 1.0 - orientation[3]  # Assuming w component should be 1
        if upright_error > 0.3:  # More than 30 degrees tilt
            return True

        # Terminate if body velocity is too high (unstable movement)
        body_velocity = np.linalg.norm(self.data.qvel[:3])
        if body_velocity > 2.0:  # Too fast movement
            return True

        # Terminate if joint velocities are too high (excessive movement)
        joint_velocities = np.linalg.norm(self.data.qvel[6:30])
        if joint_velocities > 5.0:  # Excessive joint movement
            return True

        # Check for simulation instability
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True

        return False

    def _is_stable(self):
        """Check if the robot is currently in a stable state."""
        # Check if height is close to target
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)

        # Check if body is upright
        orientation = self.data.xquat[1]
        upright_error = 1.0 - orientation[3]

        # Check if body velocity is low
        body_velocity = np.linalg.norm(self.data.qvel[:3])

        # Check if joint velocities are low
        joint_velocities = np.linalg.norm(self.data.qvel[6:30])

        # Consider stable if all conditions are met
        return (
            height_error < 0.02
            and upright_error < 0.1
            and body_velocity < 0.5
            and joint_velocities < 1.0
        )

    def _get_info(self):
        """Return training information."""
        stability_ratio = self.stable_steps / max(1, self.total_steps)
        return {
            "episode_length": self.episode_length,
            "height": self.data.qpos[2],
            "forward_velocity": self.data.qvel[0],
            "distance_traveled": self.data.qpos[0] - self.initial_x_position,
            "curriculum_stage": self.curriculum_stage,
            "stability_ratio": stability_ratio,
            "stable_steps": self.stable_steps,
            "total_steps": self.total_steps,
        }

    def set_curriculum_stage(self, stage):
        """Update curriculum stage during training."""
        self.curriculum_stage = max(1, min(3, stage))
