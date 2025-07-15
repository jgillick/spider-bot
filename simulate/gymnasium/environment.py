"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

FEMUR_INIT_POSITION = 1.0
TIBIA_INIT_POSITION = 1.25

INITIAL_JOINT_POSITIONS = (
    -1.0,  # Leg 1 - Hip
    FEMUR_INIT_POSITION,  # Leg 1 - Femur
    TIBIA_INIT_POSITION,  # Leg 1 - Tibia
    -1.0,  # Leg 2 - Hip
    FEMUR_INIT_POSITION,  # Leg 2 - Femur
    TIBIA_INIT_POSITION,  # Leg 2 - Tibia
    1.0,  # Leg 3 - Hip
    FEMUR_INIT_POSITION,  # Leg 3 - Femur
    TIBIA_INIT_POSITION,  # Leg 3 - Tibia
    1.0,  # Leg 4 - Hip
    FEMUR_INIT_POSITION,  # Leg 4 - Femur
    TIBIA_INIT_POSITION,  # Leg 4 - Tibia
    1.0,  # Leg 5 - Hip
    FEMUR_INIT_POSITION,  # Leg 5 - Femur
    TIBIA_INIT_POSITION,  # Leg 5 - Tibia
    1.0,  # Leg 6 - Hip
    FEMUR_INIT_POSITION,  # Leg 6 - Femur
    TIBIA_INIT_POSITION,  # Leg 6 - Tibia
    -1.0,  # Leg 7 - Hip
    FEMUR_INIT_POSITION,  # Leg 7 - Femur
    TIBIA_INIT_POSITION,  # Leg 7 - Tibia
    -1.0,  # Leg 8 - Hip
    FEMUR_INIT_POSITION,  # Leg 8 - Femur
    TIBIA_INIT_POSITION,  # Leg 8 - Tibia
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
        self.last_xy_body_position = None

        self.previous_action = np.zeros(24)
        self.previous_torque = np.zeros(24)
        self.feet_contact_count = 0
        self.previous_foot_contacts = np.zeros(8)  # Track foot contact changes
        self.last_foot_contacts = np.zeros(8)  # For alternation reward
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

        # Simple foot air time tracking
        self.foot_contact_history = []  # Track contact history for each foot
        self.foot_state_duration = np.zeros(
            8
        )  # How long each foot has been in current state
        self.foot_current_state = np.zeros(8)  # Current state: 0=ground, 1=air
        self.max_state_duration = 15  # Maximum steps in any state

        # Track foot air/ground time
        self.current_foot_state = np.zeros(8)
        self.current_foot_state_time = np.zeros(8)
        self.last_foot_state_time = np.zeros(8)

        # Core parameters
        self.target_height = 0.14
        self.max_torque = 8.0
        self.position_gain = 15.0
        self.velocity_gain = 0.2

        # Initialize MuJoCo environment
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(81,),
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
        self._cache_geom_ids()
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
        qpos[2] = self.target_height  # Start at target height

        # Update the state
        self.set_state(qpos, qvel)

    def step(self, action):
        """Perform a step in the environment."""
        current_foot_contacts = self._get_foot_contacts()
        self.last_foot_contacts = current_foot_contacts.copy()

        torques = self._actions_to_torques(action)
        self.last_torques = torques.copy()
        self.last_xy_body_position = self.get_body_com("Body")[:2].copy()
        self.do_simulation(torques, self.frame_skip)

        self.episode_length += 1

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
        self._initial_positions(joint_noise_scale=0.1)
        self._randomize_joint_friction()

        self.episode_length = 0
        self.total_distance = 0.0
        self.initial_x_position = self.data.qpos[0]
        self.target_height = self.data.qpos[2]
        self.last_xy_body_position = self.get_body_com("Body")[:2].copy()
        self.previous_action = np.zeros(24)  # 24 actuators
        self.feet_contact_count = 0
        self.previous_foot_contacts = np.zeros(8)  # Reset foot contact tracking
        self.last_foot_contacts = np.zeros(8)  # Reset for alternation reward
        self.previous_forward_velocity = 0.0  # Reset velocity tracking

        # Reset stability tracking
        self.stable_steps = 0
        self.total_steps = 0

        # Reset movement history
        self.position_history = []
        self.last_significant_movement_step = 0

        # Reset foot tracking
        self.foot_contact_history = []
        self.foot_state_duration = np.zeros(8)
        self.foot_current_state = np.zeros(8)

        return self._get_observation()

    def _get_observation(self):
        """Environment observations"""
        # Core observations
        velocities = self.data.qvel

        # Orientation as rotation matrix (flattened)
        orientation = self.data.xquat[1]

        # Convert joint positions back to -1.0 - 1.0 range
        positions = self.data.qpos.copy()
        for i, (low, high) in enumerate(self.joint_ranges):
            idx = FIRST_JOINT_INDEX + i
            positions[idx] = round((positions[idx] - low) / (high - low), 2)

        # Foot contacts (binary)
        foot_contacts = self._get_foot_contacts()

        observation = np.concatenate(
            [
                orientation,
                positions,
                velocities,
                foot_contacts,
                self.current_foot_state_time,
            ]
        )
        # print(f"Observable space: {len(observation)}")
        return observation

    def _calculate_rewards(self, action, torques):
        """Compute reward based on curriculum stage."""
        foot_contacts = self._get_foot_contacts()

        # Forward velocity reward (moderate scale)
        # Base reward: -4 to +4
        forward_velocity = self._get_forward_velocity()
        forward_reward = 2.0 * forward_velocity

        # Progress reward
        # Encourages movement (distance) in any direction
        # Base reward: 0 - ~0.5+
        progress_reward = 0.0
        if self.last_xy_body_position is not None:
            current_pos = self.get_body_com("Body")[:2]
            progress = np.linalg.norm(current_pos - self.last_xy_body_position)
            progress_reward = 2.5 * progress

        # Height reward (critical for staying upright)
        # Base reward: 0 - 3
        height = self.data.qpos[2]
        height_error = abs(height - self.target_height)
        height_reward = 1.5 * (1.0 - min(float(height_error), 0.1) / 0.1)

        # Upright reward (critical for stability)
        # Base reward: 0 - 2
        body_quat = self.data.xquat[1]
        w, x, y, z = body_quat
        up_vector = np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        upright_error = np.linalg.norm(up_vector - np.array([0, 0, 1]))
        upright_reward = 1.0 * (1.0 - min(float(upright_error), 0.5) / 0.5)

        # Penalty for large action changes
        # (base: 0 - 96, total: -1.92 - 0)
        if self.previous_action is not None:
            action_change = np.sum((action - self.previous_action) ** 2)
            action_smoothness_penalty = -0.02 * action_change
        else:
            action_smoothness_penalty = 0.0

        # Penalty for the robot bodies touching the ground or bodies colliding
        # Moderate penalty to prevent body ground contact
        # Base reward: -32 - 0
        contact_penalty = -0.25 * self._get_contact_penalty(foot_contacts)

        # Downward velocity penalty with cap
        # (base: 0 - 2.0, total: -2.0 - 0)
        z_velocity = self.data.qvel[2]
        downward_velocity = max(0, -z_velocity)
        downward_velocity = min(downward_velocity, 2.0)  # Cap at 2.0 m/s
        downward_velocity_penalty = -1.0 * downward_velocity

        # Simplified foot air time reward
        # Base reward: -1.25 - 1.25
        foot_air_time_reward = self._get_simplified_foot_reward(foot_contacts) * 0.125

        # Survival bonus to encourage staying upright
        survival_reward = 0.5

        # Recovery reward - encourage getting up when fallen
        # Base reward: -2.0 to +2.0
        recovery_reward = 0.0
        feet_on_ground = np.sum(foot_contacts == 1)
        if height < 0.1:  # Robot is fallen/very low
            # Reward any upward movement
            if hasattr(self, "previous_height"):
                height_change = height - self.previous_height
                if height_change > 0:
                    recovery_reward = (
                        20.0 * height_change
                    )  # Strong reward for lifting up
            # Reward for having any feet on ground (needed to push up)
            if feet_on_ground > 0:
                recovery_reward += 0.5
        self.previous_height = height

        reward = (
            forward_reward
            + height_reward
            + upright_reward
            + action_smoothness_penalty
            + contact_penalty
            + downward_velocity_penalty
            + foot_air_time_reward
            + progress_reward
            + survival_reward
            + recovery_reward
        )

        return reward

    def _get_simplified_foot_reward(self, foot_contacts):
        """
        Simplified foot reward focusing on basic gait patterns.
        Returns value in range [-10, 10] before external scaling.
        """
        # Track foot state changes
        for i, state in enumerate(foot_contacts):
            if state != self.current_foot_state[i]:
                self.current_foot_state[i] = state
                self.last_foot_state_time[i] = self.current_foot_state_time[i]
                self.current_foot_state_time[i] = 0
            self.current_foot_state_time[i] += 1

        # Basic gait balance reward - encourage some feet in air, some on ground
        feet_on_ground = np.sum(foot_contacts == 1)

        # Optimal is 4-5 feet on ground
        if 4 <= feet_on_ground <= 5:
            gait_balance_reward = 5.0
        elif feet_on_ground == 6:
            gait_balance_reward = 2.0
        elif feet_on_ground == 7:
            gait_balance_reward = 0.0
        else:
            # All feet in same state - bad
            gait_balance_reward = -5.0

        # Simple alternation reward - reward switching feet
        foot_switches = np.sum(foot_contacts != self.last_foot_contacts)
        alternation_reward = min(foot_switches * 0.5, 3.0)  # Cap at 3.0

        # Diagonal gait bonus (spider-like walking pattern)
        diagonal_pairs = [
            (foot_contacts[0] and foot_contacts[5]),  # Leg L1 & Leg R2
            (foot_contacts[1] and foot_contacts[4]),  # Leg L2 & Leg R1
            (foot_contacts[2] and foot_contacts[7]),  # Leg L3 & Leg R4
            (foot_contacts[3] and foot_contacts[6]),  # Leg L4 & Leg R3
        ]
        diagonal_bonus = sum(diagonal_pairs) * 0.5

        total_reward = gait_balance_reward + alternation_reward + diagonal_bonus
        return total_reward

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

    def _randomize_joint_friction(self):
        """Add random damping and friction loss to all joints except the free joint."""
        # Reasonable ranges for robot joints
        friction_range = (0.001, 0.05)
        damping_range = (0.0, 0.1)

        # Generate random damping and friction loss
        random_damping = self.np_random.uniform(damping_range[0], damping_range[1])
        random_friction = self.np_random.uniform(friction_range[0], friction_range[1])

        # Apply to all actuated joints
        first_index = len(self.model.dof_damping) - DOF
        for i in range(DOF):
            joint_id = first_index + i
            self.model.dof_damping[joint_id] = random_damping
            self.model.dof_frictionloss[joint_id] = random_friction

    def _get_foot_air_time_reward(self, foot_contacts):
        """
        Calculate reward for proper walking gait with feet alternating between air and ground.
        Base reward: -10 - 10
        """

        # Track how long each foot has been in the current state
        # States: 1 = ground, 0 = air
        for i, state in enumerate(foot_contacts):
            if state != self.current_foot_state[i]:
                self.current_foot_state[i] = state
                self.last_foot_state_time[i] = self.current_foot_state_time[i]
                self.current_foot_state_time[i] = 0
            self.current_foot_state_time[i] += 1

        # Reward based on an even distribution of foot steps value
        # This encourages all feet to have similar air/ground time durations
        # Base reward: 0 - 1
        distribution_reward = 0.0
        if np.any(self.last_foot_state_time > 0):
            # Filter out feet that have not completed a step yet
            mean_air_time = np.mean(
                self.last_foot_state_time[self.last_foot_state_time > 0]
            )
            if mean_air_time > 0:
                # Calculate coefficient of variation (std/mean) - lower is better
                # Reward for low variation (even distribution)
                std_air_time = np.std(
                    self.last_foot_state_time[self.last_foot_state_time > 0]
                )
                cv = std_air_time / mean_air_time
                distribution_reward = 4.0 * np.exp(-3.0 * cv)

        # Balanced air/ground reward - encourage proper gait
        # Base reward: -2 - 4
        feet_in_air = np.sum(foot_contacts == 0)
        feet_on_ground = np.sum(foot_contacts == 1)
        if feet_in_air > 0 and feet_on_ground > 0:
            gait_balance = min(feet_in_air, feet_on_ground)
            air_time_reward = 1.0 * gait_balance
        else:
            # Penalty for all feet in same state
            air_time_reward = -2.0

        # Bell curve reward for optimal foot state duration
        # Encourage feet to spend optimal time in each state (not too short, not too long)
        # Base reward: -4 - 4
        OPTIMAL_STATE_TIME = 40.0  # Optimal duration in each state
        STATE_TIME_TOLERANCE = 15.0  # Acceptable range around optimal
        foot_state_reward = 0.0
        for foot_time in self.current_foot_state_time:
            # Only consider feet that have been in a state
            if foot_time > 0:
                time_diff = abs(foot_time - OPTIMAL_STATE_TIME)

                # Bell curve: reward peaks at optimal time, decreases as we move away
                if time_diff <= STATE_TIME_TOLERANCE:
                    normalized_diff = time_diff / STATE_TIME_TOLERANCE
                    bell_reward = 4.0 * np.exp(-2.0 * normalized_diff**2)
                    foot_state_reward += bell_reward
                else:
                    # Outside tolerance - penalty
                    over_tolerance = time_diff - STATE_TIME_TOLERANCE
                    penalty = max(-0.5 * np.exp(0.1 * over_tolerance), 4.0)
                    foot_state_reward += penalty
        foot_state_reward /= 8.0  # averaged across all feet

        total_reward = distribution_reward + air_time_reward + foot_state_reward
        return total_reward

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

    def _cache_geom_ids(self):
        """Cache geom IDs by looking up their names."""
        self.foot_geom_ids = []
        self.tibia_geom_ids = []
        self.ground_plane_id = self.model.geom("floor").id

        # Foot geoms
        for leg_num in range(1, 9):  # Legs 1-8
            foot_name = f"Leg{leg_num}_Tibia_Foot"
            foot_geom = self.model.geom(foot_name)

            if foot_geom is not None:
                self.foot_geom_ids.append(foot_geom.id)
            else:
                raise Exception(f"Could not find geom '{foot_name}'")

            tibia_name = f"Leg{leg_num}_Tibia_Leg"
            tibia_geom = self.model.geom(tibia_name)
            if tibia_geom is not None:
                self.tibia_geom_ids.append(tibia_geom.id)
            else:
                raise Exception(f"Warning: Could not find geom '{tibia_name}'")

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

    def _get_contact_penalty(self, foot_contacts):
        """Penalize contact with non-foot parts of the robot."""
        penalty = 0.0

        # Check each contact in the simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            ground_contact = (
                contact.geom1 == self.ground_plane_id
                or contact.geom2 == self.ground_plane_id
            )
            foot_contact = (
                contact.geom1 in self.foot_geom_ids
                or contact.geom2 in self.foot_geom_ids
            )
            tibia_contact = (
                contact.geom1 in self.tibia_geom_ids
                or contact.geom2 in self.tibia_geom_ids
            )

            # Ignore tibia contact if this foot is also on the ground
            # Sometimes the foot sinks below the ground plane and the tibia makes contact
            if tibia_contact and ground_contact:
                tibia_geom_id = (
                    contact.geom1
                    if contact.geom1 in self.tibia_geom_ids
                    else contact.geom2
                )
                tibia_index = self.tibia_geom_ids.index(tibia_geom_id)
                if foot_contacts[tibia_index] == 1:
                    continue

            # Only penalize if it's not a foot on ground contact
            if not (foot_contact and ground_contact):
                if ground_contact:
                    # Penalty for any other body on the ground
                    penalty += 2.0
                else:
                    # Mild penalty for robot part collision
                    penalty += 0.2

        return penalty

    def _get_gyro_stability(self):
        """Get gyro sensor stability measurement."""
        # Try to find gyro sensor by iterating through sensors
        gyro_sensor_id = None
        for i in range(self.model.nsensor):
            if self.model.sensor(i).name == "gyro_sensor":
                gyro_sensor_id = i
                break

        if gyro_sensor_id is not None:
            gyro_data = self.data.sensordata[gyro_sensor_id : gyro_sensor_id + 3]
            angular_velocity_magnitude = np.linalg.norm(gyro_data)

            # Convert to stability score (0 = unstable, 1 = very stable)
            # Use exponential decay: exp(-angular_velocity)
            stability = np.exp(-2.0 * angular_velocity_magnitude)

            return stability
        else:
            # Fallback if gyro sensor not found
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
            has_foot_contact = (
                contact.geom1 in self.foot_geom_ids
                or contact.geom2 in self.foot_geom_ids
            )
            has_ground_contact = (
                contact.geom1 == self.ground_plane_id
                or contact.geom2 == self.ground_plane_id
            )

            if has_foot_contact and has_ground_contact:
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
