"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from mujoco import mjtObj, mj_name2id

INITIAL_JOINT_POSITIONS = (
    -1.0,  # Leg 1 - Hip
    0.75,  # Leg 1 - Femur
    1.0,  # Leg 1 - Tibia
    -1.0,  # Leg 2 - Hip
    0.75,  # Leg 2 - Femur
    1.0,  # Leg 2 - Tibia
    1.0,  # Leg 3 - Hip
    0.75,  # Leg 3 - Femur
    1.0,  # Leg 3 - Tibia
    1.0,  # Leg 4 - Hip
    0.75,  # Leg 4 - Femur
    1.0,  # Leg 4 - Tibia
    1.0,  # Leg 5 - Hip
    0.75,  # Leg 5 - Femur
    1.0,  # Leg 5 - Tibia
    1.0,  # Leg 6 - Hip
    0.75,  # Leg 6 - Femur
    1.0,  # Leg 6 - Tibia
    -1.0,  # Leg 7 - Hip
    0.75,  # Leg 7 - Femur
    1.0,  # Leg 7 - Tibia
    -1.0,  # Leg 8 - Hip
    0.75,  # Leg 8 - Femur
    1.0,  # Leg 8 - Tibia
)

DOF = 24
FIRST_JOINT_INDEX = 7


class SpiderRobotEnv(MujocoEnv):
    """
    Simplified environment with curriculum learning and cleaner reward structure.
    """

    def __init__(
        self,
        xml_file,
        frame_skip=5,
        render_mode=None,
        **kwargs,
    ):
        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = 0.0

        self.previous_action = np.zeros(24)
        self.previous_torque = np.zeros(24)
        self.feet_contact_count = 0
        self.previous_foot_contacts = np.zeros(8)  # Track foot contact changes
        self.previous_forward_velocity = 0.0  # Track velocity changes

        # Episode stability tracking
        self.stable_steps = 0
        self.total_steps = 0

        # Movement history tracking for continuous walking rewards
        self.movement_history_window = 50  # Track last 50 steps
        self.position_history = []  # Will store recent x positions
        self.last_significant_movement_step = (
            0  # Track when we last moved significantly
        )

        # Core parameters
        self.target_height = 0.134
        self.max_torque = 8.0
        self.position_gain = 15.0
        self.velocity_gain = 0.8

        # Initialize MuJoCo environment
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(74,),
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

        # Cache model data
        self._cache_foot_geom_ids()
        self._cache_joint_ranges()

        # Set initial joint positions
        self._initial_positions()

    def _initial_positions(self, joint_noise_scale=0.0):
        """Set the initial joint positions from the constant."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Create random join noise
        joint_perturbations = self.np_random.uniform(
            low=-joint_noise_scale,
            high=joint_noise_scale,
            size=len(INITIAL_JOINT_POSITIONS),
        )

        # Set initial joint positions from the constant
        for i, pos_str in enumerate(INITIAL_JOINT_POSITIONS):
            idx = FIRST_JOINT_INDEX + i
            qpos[idx] = float(pos_str)
            if joint_noise_scale > 0.0:
                qpos[idx] += joint_perturbations[i]
            qpos[idx] = round(qpos[idx], 2)

        # Ensure the robot starts at a reasonable height
        # Set the z-position (height) to be above ground
        qpos[2] = 0.134

        # Update the state
        self.set_state(qpos, qvel)

    def _actions_to_torques(self, action):
        """
        Convert actions to torques using PD control
        """
        # Scale actions to joint ranges
        target_positions = []
        for i, (low, high) in enumerate(self.joint_ranges):
            scaled_pos = low + (action[i] + 1.0) * 0.5 * (high - low)
            target_positions.append(scaled_pos)
        target_positions = np.array(target_positions)

        # PD control
        current_positions = self.data.qpos[-DOF:]
        current_velocities = self.data.qvel[-DOF:]

        # Calculate torques with safety checks
        position_errors = target_positions - current_positions
        velocity_errors = current_velocities

        # Clip position & velocity errors to prevent extreme corrections
        position_errors = np.clip(position_errors, -0.5, 0.5)
        velocity_errors = np.clip(velocity_errors, -5.0, 5.0)

        torques = (
            self.position_gain * position_errors - self.velocity_gain * velocity_errors
        )
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        return torques

    def step(self, action):
        """Perform a step in the environment."""
        torques = self._actions_to_torques(action)
        self.last_xy_body_position = self.get_body_com("Body")[:2].copy()
        self.do_simulation(torques, self.frame_skip)

        observation = self._get_observation()
        reward = self._calculate_rewards(action, torques)
        terminated = self._is_terminated()
        truncated = False
        info = {}

        self.previous_action = action.copy()
        self.previous_torque = torques.copy()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Reset with curriculum-appropriate randomization."""
        self._initial_positions(joint_noise_scale=0.5)

        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = self.data.qpos[0]
        self.target_height = self.data.qpos[2]
        self.previous_action = np.zeros(24)  # 24 actuators
        self.feet_contact_count = 0
        self.previous_foot_contacts = np.zeros(8)  # Reset foot contact tracking
        self.previous_forward_velocity = 0.0  # Reset velocity tracking

        # Reset stability tracking
        self.stable_steps = 0
        self.total_steps = 0

        # Reset movement history
        self.position_history = []
        self.last_significant_movement_step = 0

        return self._get_observation()

    def _get_observation(self):
        """Environment observations"""
        # Core observations
        body_height = self.data.qpos[2]
        velocities = self.data.qvel

        # Orientation as rotation matrix (flattened)
        orientation = self.data.xquat[1]

        # Convert joint positions back to -1.0 - 1.0 range
        joint_positions = self.data.qpos.copy()
        for i, (low, high) in enumerate(self.joint_ranges):
            idx = FIRST_JOINT_INDEX + i
            joint_positions[idx] = round((joint_positions[idx] - low) / (high - low), 2)

        # Foot contacts (binary)
        foot_contacts = self._get_foot_contacts()

        observation = np.concatenate(
            [
                [body_height],
                orientation,
                joint_positions,
                velocities,
                foot_contacts,
            ]
        )
        # print(f"Observable space: {len(observation)}")
        return observation

    def _calculate_rewards(self, action, torques):
        """Compute reward based on curriculum stage."""

        # Forward reward (base: 2, -2, total: -12 - 12)
        forward_velocity = self._get_forward_velocity()
        forward_reward = 6.0 * forward_velocity

        # Height reward (base: 0 - 1, total: 0 - 2)
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)
        height_reward = 2.0 * np.exp(-30 * height_error**2)

        # Simple gait reward (base: 0 - 8, total: 0 - 10)
        foot_contacts = self._get_foot_contacts()
        num_feet_on_ground = np.sum(foot_contacts)
        foot_contact_reward = 10.0 if num_feet_on_ground >= 4 else 0.0

        # Upright reward (base: 0 - 1, total: 0 - 2)
        # Convert quaternion to rotation matrix and extract up vector
        # For a quaternion [w, x, y, z], the up vector is the third column of rotation matrix
        body_quat = self.data.xquat[1]
        w, x, y, z = body_quat
        up_vector = np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        # Perfectly upright means up_vector should be [0, 0, 1]
        upright_error = np.linalg.norm(up_vector - np.array([0, 0, 1]))
        upright_reward = 2 * np.exp(-8 * upright_error**2)

        # Penalty for large action changes (base: 0 - 96, total: -9.6 - 0)
        if self.previous_action is not None:
            action_change = np.sum((action - self.previous_action) ** 2)
            action_smoothness_penalty = -0.1 * action_change
        else:
            action_smoothness_penalty = 0.0

        # Penalty for actuator direction changes (base: 0 - 24, total: -7 - 0)
        # direction_change_penalty = 0.0
        # if self.previous_torque is not None:
        #     sign_flips = np.sign(self.previous_torque) != np.sign(torques)
        #     nonzero_flip = (self.previous_torque != 0) & (torques != 0)
        #     actual_flips = sign_flips & nonzero_flip
        #     num_flips = np.sum(actual_flips)
        #     direction_change_penalty = -0.3 * num_flips

        # Control cost
        # (base: 0 - 1536, total: -11 - 0)
        # control_cost = -0.007 * np.sum(np.square(torques))
        # (base: 0 - 24, total: -4.8 - 0)
        control_cost = -0.2 * np.sum(np.square(action))

        # Penalty for the robot touching the ground or bodies colliding
        # (base: 0 - 9, total: -9 - 0)
        contact_penalty = -1 * self._get_contact_penalty()

        # Downward velocity penalty (penalize falling)
        # (base: 0 - inf, typical: -20 - 0, extreme falls: -45 - 0)
        z_velocity = self.data.qvel[2]
        downward_velocity = max(0, -z_velocity)
        # Quadratic penalty that increases with falling speed
        # Scale factor of 5.0 makes a fall of 1 m/s cost -5, 2 m/s cost -20
        downward_velocity_penalty = -5.0 * (downward_velocity**2)

        reward = (
            forward_reward
            + height_reward
            + upright_reward
            + foot_contact_reward
            + control_cost
            + action_smoothness_penalty
            + contact_penalty
            + downward_velocity_penalty
            # + direction_change_penalty
        )
        return reward

    def _get_forward_velocity(self):
        """
        Get the forward velocity of the robot, regarless of world orientation.
        """
        # Body world position and XY velocity
        xy_body_position = self.get_body_com("Body")[:2].copy()
        xy_velocity = (xy_body_position - self.last_xy_body_position) / self.dt

        # Orientation and forward direction
        # Convert quaternion to yaw angle (rotation around z-axis)
        body_quat = self.data.xquat[1]
        yaw = np.arctan2(
            2 * (body_quat[0] * body_quat[3] + body_quat[1] * body_quat[2]),
            1 - 2 * (body_quat[2] ** 2 + body_quat[3] ** 2),
        )

        # Robot's forward direction in world coordinates
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])

        # Put it all together
        forward_velocity = np.dot(xy_velocity, forward_vec)

        return forward_velocity

    def _cache_foot_geom_ids(self):
        """Cache foot geom IDs by looking up their names."""
        self.foot_geom_ids = []

        # Look for foot geoms by name pattern "LegX_Tibia_foot"
        for leg_num in range(1, 9):  # Legs 1-8
            foot_name = f"Leg{leg_num}_Tibia_Foot_geom"
            foot_id = mj_name2id(self.model, mjtObj.mjOBJ_GEOM, foot_name)
            if foot_id is not None:
                self.foot_geom_ids.append(foot_id)
            else:
                print(f"⚠️ Warning: Could not find geom '{foot_name}'")

    def _cache_joint_ranges(self):
        """Cache the join position ranges."""
        self.joint_ranges = []
        for i in range(0, self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if self.model.jnt_limited[joint_id]:
                self.joint_ranges.append(
                    (
                        self.model.jnt_range[joint_id, 0],
                        self.model.jnt_range[joint_id, 1],
                    )
                )
            else:
                self.joint_ranges.append((-np.pi, np.pi))

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
                    penalty += 2.0
                else:
                    # Very mild penalty for robot parts touching each other
                    penalty += 0.1

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
            if (
                contact.geom1 in self.foot_geom_ids
                or contact.geom2 in self.foot_geom_ids
            ):
                foot_geom_id = (
                    contact.geom1
                    if contact.geom1 in self.foot_geom_ids
                    else contact.geom2
                )
                foot_idx = self.foot_geom_ids.index(foot_geom_id)
                contacts[foot_idx] = 1.0

        return contacts

    def _is_terminated(self):
        """Check if episode should terminate (robot has fallen or become unstable)."""

        # Check for NaN values (simulation failure)
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print(
                f"🔴 Episode terminated at step {self.episode_length}: NaN values detected"
            )
            return True

        # Check if robot has flipped over
        body_quat = self.data.xquat[1]
        w, x, y, z = body_quat
        up_vector = np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        upright_error = np.linalg.norm(up_vector - np.array([0, 0, 1]))
        if upright_error > 0.5:
            print(
                f"🔴 Episode terminated at step {self.episode_length}: robot has flipped"
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
