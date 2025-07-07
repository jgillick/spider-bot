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
        self.previous_action = np.zeros(24)  # 24 actuators
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
            shape=(48,),  # Updated observation space size
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
            shape=(24,),  # Normalized actions (24 actuators)
            dtype=np.float32,
        )

        # Cache foot geom IDs by name
        self._cache_foot_geom_ids()

        # Set initial joint positions
        self._set_initial_joint_positions()

    def _set_initial_joint_positions(self):
        """Set the initial joint positions from the constant."""
        qpos = self.data.qpos.copy()

        # Set initial joint positions from the constant
        # Skip the free joint (qpos[0:7]) and start at the first actuated joint (qpos[7])
        for i, pos_str in enumerate(INITIAL_JOINT_POSITIONS):
            if i < len(qpos) - 7:  # Ensure we don't go out of bounds
                qpos[7 + i] = float(pos_str)  # Start at index 7 (first actuated joint)

        # Ensure the robot starts at a reasonable height
        # Set the z-position (height) to be above ground
        qpos[2] = 0.15  # Start 15cm above ground

        # Update the state
        self.set_state(qpos, self.data.qvel)

        # Let the physics settle for a few steps
        for _ in range(10):
            self.do_simulation(np.zeros(24), 1)

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
        # Use exact number of actuated joints (24)
        current_positions = self.data.qpos[
            7:31
        ]  # Start at index 7, get 24 actuated joints (indices 7-30)
        current_velocities = self.data.qvel[
            6:30
        ]  # Start at index 6, get 24 actuated joint velocities (indices 6-29)

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
        """Reset with curriculum-appropriate randomization."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Stage 1: 30% chance to start from completely random pose
        if self.curriculum_stage == 1 and self.np_random.random() < 0.3:
            # Complete random initialization for Stage 1
            # Random joint positions within limits
            for i in range(24):  # 24 actuated joints
                joint_id = i + 1  # Joint IDs start at 1 (0 is free joint)
                if self.model.jnt_limited[joint_id]:
                    low = self.model.jnt_range[joint_id, 0]
                    high = self.model.jnt_range[joint_id, 1]
                    qpos[7 + i] = self.np_random.uniform(
                        low * 0.8, high * 0.8
                    )  # 80% of range for safety
                else:
                    qpos[7 + i] = self.np_random.uniform(-1.5, 1.5)

            # Random but reasonable height
            qpos[2] = self.np_random.uniform(0.10, 0.20)  # 10-20cm

            # Random but mostly upright orientation
            roll_angle = self.np_random.uniform(-0.2, 0.2)  # ±0.2 rad
            pitch_angle = self.np_random.uniform(-0.2, 0.2)
            qpos[3] = np.cos(roll_angle / 2) * np.cos(pitch_angle / 2)
            qpos[4] = np.sin(roll_angle / 2) * np.cos(pitch_angle / 2)
            qpos[5] = np.cos(roll_angle / 2) * np.sin(pitch_angle / 2)
            qpos[6] = 0
            quat_norm = np.linalg.norm(qpos[3:7])
            qpos[3:7] /= quat_norm

        else:
            # Normal initialization with INITIAL_JOINT_POSITIONS
            # Set initial joint positions from the constant
            for i, pos_str in enumerate(INITIAL_JOINT_POSITIONS):
                if i < len(qpos) - 7:
                    qpos[7 + i] = float(pos_str)

            # Ensure the robot starts at a reasonable height
            qpos[2] = 0.15  # Start 15cm above ground

            # Keep body orientation upright
            qpos[3:7] = [1, 0, 0, 0]  # Quaternion for upright orientation

        # Curriculum-based randomization
        if self.curriculum_stage == 1:
            # Stage 1: Significant randomization to learn recovery
            # Add random perturbations to joint positions
            joint_noise_scale = 0.3  # ±0.3 radians (~17 degrees)
            joint_perturbations = self.np_random.uniform(
                low=-joint_noise_scale,
                high=joint_noise_scale,
                size=len(INITIAL_JOINT_POSITIONS),
            )
            for i in range(min(len(joint_perturbations), len(qpos) - 7)):
                qpos[7 + i] += joint_perturbations[i]

            # Randomize body orientation slightly
            # Small rotations around roll and pitch axes
            roll_angle = self.np_random.uniform(-0.1, 0.1)  # ±0.1 rad (~6 degrees)
            pitch_angle = self.np_random.uniform(-0.1, 0.1)

            # Convert to quaternion (approximate for small angles)
            qpos[3] = np.cos(roll_angle / 2) * np.cos(pitch_angle / 2)  # w
            qpos[4] = np.sin(roll_angle / 2) * np.cos(pitch_angle / 2)  # x
            qpos[5] = np.cos(roll_angle / 2) * np.sin(pitch_angle / 2)  # y
            qpos[6] = 0  # z (no yaw variation)

            # Normalize quaternion
            quat_norm = np.linalg.norm(qpos[3:7])
            qpos[3:7] /= quat_norm

            # Randomize initial height slightly
            qpos[2] += self.np_random.uniform(-0.02, 0.02)  # ±2cm variation

        else:
            # Stage 2 & 3: Less randomization (robot should be more stable)
            joint_noise_scale = 0.1  # ±0.1 radians (~6 degrees)
            joint_perturbations = self.np_random.uniform(
                low=-joint_noise_scale,
                high=joint_noise_scale,
                size=len(INITIAL_JOINT_POSITIONS),
            )
            for i in range(min(len(joint_perturbations), len(qpos) - 7)):
                qpos[7 + i] += joint_perturbations[i]

        # Zero all velocities for stable start
        qvel[:] = 0.0

        self.set_state(qpos, qvel)

        # Let physics settle for a few steps
        for _ in range(5):
            self.do_simulation(np.zeros(24), 1)

        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = self.data.qpos[0]
        self.target_height = self.data.qpos[2]
        self.previous_action = np.zeros(24)  # 24 actuators
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
        joint_positions = self.data.qpos[
            7:31
        ]  # Start at index 7, get 24 actuated joints (indices 7-30)
        joint_velocities = self.data.qvel[
            6:30
        ]  # Start at index 6, get 24 actuated joint velocities (indices 6-29)

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

        return obs[:48]  # Ensure correct size (1+3+4+24+24+8 = 64, but we want 48)

    def _compute_curriculum_reward(self, action):
        """Compute reward based on curriculum stage."""
        # Common measurements
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)

        forward_velocity = self.data.qvel[0]
        lateral_velocity = abs(self.data.qvel[1])

        # Calculate upright orientation properly
        # Get body's up vector (z-axis) in world coordinates
        body_quat = self.data.xquat[1]  # Body quaternion
        # Convert quaternion to rotation matrix and extract up vector
        # For a quaternion [w, x, y, z], the up vector is the third column of rotation matrix
        w, x, y, z = body_quat
        up_vector = np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        # Perfectly upright means up_vector should be [0, 0, 1]
        # Calculate error as distance from ideal up vector
        upright_error = np.linalg.norm(up_vector - np.array([0, 0, 1]))

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
            joint_velocities = self.data.qvel[
                6:30
            ]  # Start at index 6, get 24 actuated joint velocities (indices 6-29)
            joint_movement_penalty = -0.5 * np.sum(np.abs(joint_velocities))

            # Reward for minimal linear body movement (not angular - that's handled by gyro)
            linear_velocity = self.data.qvel[:3]  # Only x, y, z velocities
            body_stillness = 5.0 * np.exp(-10 * np.sum(linear_velocity**2))

            gyro_reward = 6.0 * gyro_stability  # Stronger gyro reward

            # Bonus for maintaining target height
            height_bonus = 2.0 if height_error < 0.005 else 0.0

            # Penalty for action changes (encourage smooth, minimal actions)
            if self.previous_action is not None:
                action_change = np.sum((action - self.previous_action) ** 2)
                action_smoothness_penalty = -0.3 * action_change
            else:
                action_smoothness_penalty = 0.0

            # Foot contact reward for Stage 1 - encourage all feet on ground
            foot_contacts = self._get_foot_contacts()
            num_feet_on_ground = np.sum(foot_contacts)
            # Reward having 6-8 feet on ground (some flexibility for natural weight shifting)
            foot_contact_reward = (
                3.0 * (num_feet_on_ground / 8.0) if num_feet_on_ground >= 6 else 0.0
            )

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
                + foot_contact_reward  # New reward component
                + contact_penalty * 0.2
                + 2.0  # Base reward
            ) * stability_multiplier

        # Stage 2: Movement (add forward progress)
        elif self.curriculum_stage == 2:
            height_reward = 3.0 * np.exp(-30 * height_error**2)
            upright_reward = 2.0 * np.exp(-8 * upright_error**2)

            # Progressive forward reward (encourage consistent forward motion)
            forward_reward = 15.0 * np.clip(forward_velocity, 0, 0.5)

            # Penalty for NOT moving forward (anti-freezing)
            movement_penalty = -3.0 if forward_velocity < 0.05 else 0.0

            # Penalize sideways drift
            lateral_penalty = -3.0 * lateral_velocity  # Reduced from -5.0

            # Gait coordination reward
            foot_contacts = self._get_foot_contacts()
            alternating_contacts = self._get_gait_coordination_reward(foot_contacts)

            # Bonus for lifting feet (to encourage stepping)
            num_feet_lifted = 8 - np.sum(foot_contacts)
            stepping_bonus = 1.0 * min(
                float(num_feet_lifted) / 4.0, 1.0
            )  # Max when 4 feet lifted

            gyro_reward = (
                1.5 * gyro_stability
            )  # Reduced from 2.0 to allow more movement

            # Distance traveled bonus
            distance_traveled = self.data.qpos[0] - self.initial_x_position
            distance_bonus = 0.5 * distance_traveled  # Small cumulative bonus

            reward = (
                height_reward
                + upright_reward
                + forward_reward
                + movement_penalty
                + lateral_penalty
                + alternating_contacts
                + stepping_bonus
                + gyro_reward
                + distance_bonus
                + contact_penalty * 0.1  # Further reduced
                + 1.0  # Reduced base reward
            )

        # Stage 3: Efficiency (optimize everything)
        else:
            # Maintain stability but with less emphasis
            height_reward = 2.5 * np.exp(
                -20 * height_error**2
            )  # Increased from 2.0, relaxed exp
            upright_reward = 2.0 * np.exp(
                -6 * upright_error**2
            )  # Increased from 1.5, relaxed exp

            # SIGNIFICANTLY increase forward motion reward with progressive scaling
            # Stage 3 should reward speed much more than Stage 2
            forward_reward = 20.0 * np.clip(
                forward_velocity, 0, 0.8
            )  # Increased from 5.0 to 20.0

            # Add acceleration bonus to encourage pushing speed limits
            if hasattr(self, "previous_forward_velocity"):
                acceleration = forward_velocity - self.previous_forward_velocity
                acceleration_bonus = 2.0 * np.clip(
                    acceleration, 0, 0.1
                )  # Reward positive acceleration
            else:
                acceleration_bonus = 0.0

            gyro_reward = 2.0 * gyro_stability  # Increased from 1.5 for better balance

            # Energy efficiency (smoother actions) - reduced penalty
            if self.previous_action is not None:
                action_change = np.sum((action - self.previous_action) ** 2)
                smoothness_reward = 0.5 * np.exp(
                    -0.3 * action_change
                )  # Reduced weight and decay
            else:
                smoothness_reward = 0.0

            # Advanced gait quality - increased importance
            foot_contacts = self._get_foot_contacts()
            gait_reward = (
                self._get_gait_coordination_reward(foot_contacts) * 1.5
            )  # Increased from 0.8

            # Speed consistency - less strict to allow acceleration
            if hasattr(self, "previous_forward_velocity"):
                speed_consistency = 0.5 * np.exp(
                    -1.0
                    * (forward_velocity - self.previous_forward_velocity)
                    ** 2  # Relaxed from -3.0
                )
            else:
                speed_consistency = 0.0
            self.previous_forward_velocity = forward_velocity

            # Distance milestone bonuses
            distance_traveled = self.data.qpos[0] - self.initial_x_position
            distance_bonus = 1.0 * distance_traveled  # Progressive distance reward

            # Milestone rewards for reaching certain distances
            if distance_traveled > 2.0:
                distance_bonus += 5.0
            if distance_traveled > 5.0:
                distance_bonus += 10.0

            reward = (
                height_reward
                + upright_reward
                + forward_reward
                + acceleration_bonus
                + gyro_reward
                + smoothness_reward
                + gait_reward
                + speed_consistency
                + distance_bonus
                + contact_penalty * 0.05  # Further reduced from 0.1
                + 2.0  # Increased base reward from 1.0
            )

        return reward

    def _cache_foot_geom_ids(self):
        """Cache foot geom IDs by looking up their names."""
        self.foot_geom_ids = set()

        # Look for foot geoms by name pattern "LegX_Tibia_foot"
        for leg_num in range(1, 9):  # Legs 1-8
            foot_name = f"Leg{leg_num}_Tibia_Foot_geom"
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
            gyro_data = self.data.sensordata[gyro_sensor_id : gyro_sensor_id + 3]
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

        # Very minimal termination conditions - let the robot learn from any state

        # Only terminate on complete simulation failure
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print(
                f"🔴 Episode terminated at step {self.episode_length}: NaN values detected"
            )
            return True

        # Only terminate if robot is completely underground (simulation error)
        height = self.data.qpos[2]
        if height < -0.1:  # Only if completely underground
            print(
                f"🔴 Episode terminated at step {self.episode_length}: Robot underground (height: {height:.4f})"
            )
            return True

        return False

    def _is_stable(self):
        """Check if the robot is currently in a stable state."""
        # Check if height is close to target
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)

        # Check if body is upright (using same calculation as reward)
        body_quat = self.data.xquat[1]
        w, x, y, z = body_quat
        up_vector = np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        upright_error = np.linalg.norm(up_vector - np.array([0, 0, 1]))

        # Check if body velocity is low
        body_velocity = np.linalg.norm(self.data.qvel[:3])

        # Check if joint velocities are low
        joint_velocities = np.linalg.norm(
            self.data.qvel[6:30]
        )  # Start at index 6, get 24 actuated joint velocities (indices 6-29)

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

        # Add termination reason for debugging
        termination_reason = "running"
        if self._is_terminated():
            height = self.data.qpos[2]
            orientation = self.data.xquat[1]
            upright_error = 1.0 - orientation[3]
            body_velocity = np.linalg.norm(self.data.qvel[:3])
            joint_velocities = np.linalg.norm(
                self.data.qvel[6:30]
            )  # Start at index 6, get 24 actuated joint velocities (indices 6-29)

            if height < 0.03:
                termination_reason = "height_too_low"
            elif upright_error > 0.5:
                termination_reason = "body_tilted"
            elif body_velocity > 5.0:
                termination_reason = "body_velocity_high"
            elif joint_velocities > 10.0:
                termination_reason = "joint_velocities_high"
            elif np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
                termination_reason = "nan_detected"

        return {
            "episode_length": self.episode_length,
            "height": self.data.qpos[2],
            "forward_velocity": self.data.qvel[0],
            "distance_traveled": self.data.qpos[0] - self.initial_x_position,
            "curriculum_stage": self.curriculum_stage,
            "stability_ratio": stability_ratio,
            "stable_steps": self.stable_steps,
            "total_steps": self.total_steps,
            "termination_reason": termination_reason,
        }

    def set_curriculum_stage(self, stage):
        """Update curriculum stage during training."""
        self.curriculum_stage = max(1, min(3, stage))
