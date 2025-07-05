"""
Spider Robot Reinforcement Learning Environment and Training
Uses Gymnasium and MuJoCo for physics simulation
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor


class SpiderRobotEnv(MujocoEnv):
    """
    Custom Gymnasium environment for training a spider robot to walk.

    The robot has:
    - 8 legs
    - 3 actuators per leg (24 total actuators)
    - IMU sensor for orientation/acceleration feedback
    - Position-controlled torque motors with limits
    """

    def __init__(
        self,
        xml_file,
        frame_skip=5,
        render_mode=None,
        camera_name="bodycam",
        debug=False,
        **kwargs,
    ):
        # Initialize tracking variables
        self.previous_x_position = 0.0
        self.previous_hip_positions = {}
        self.previous_foot_positions = {}
        self.step_lengths = []
        self.max_step_history = 50

        # Contact tracking
        self.contact_history = []
        self.max_contact_history = 20

        # Joint smoothness tracking
        self.previous_joint_velocities = None
        self.joint_acceleration_history = []
        self.max_acceleration_history = 10

        # Leg grouping for gait coordination
        self.leg_groups = {
            "group_a": [0, 2, 4, 6],  # Front-left, back-left, front-right, back-right
            "group_b": [1, 3, 5, 7],  # Front-right, back-right, front-left, back-left
        }

        # Debug mode
        self.debug = debug
        self.steps_taken = 0

        # Gait phase tracking
        self.gait_phase = 0.0
        self.gait_frequency = 0.5  # Hz

        # Control parameters - reduced gains for stability
        self.max_torque = 8.0
        self.position_gain = 10.0  # Reduced from 20.0
        self.velocity_gain = 0.5  # Reduced from 1.0

        # Initialize MuJoCo environment
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_obs_dim(xml_file),),
            dtype=np.float64,
        )

        # Set metadata with correct render_fps
        # Use a default render_fps since we can't easily get it from the model before initialization
        render_fps = 100  # Default to 100 FPS

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": render_fps,
        }

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            **kwargs,
        )

        # Get body ID after MujocoEnv initialization
        # Try to find torso body, otherwise use body 0 (usually the root)
        self.torso_id = 0
        for i in range(self.model.nbody):
            if self.model.body(i).name == "Body":
                self.torso_id = i
                break

        # Find foot geom IDs
        self.foot_geom_ids = []
        for i in range(1, 9):  # 8 legs
            foot_name = f"Leg{i}_Tibia_foot_site"
            for j in range(self.model.ngeom):
                if self.model.geom(j).name == foot_name:
                    self.foot_geom_ids.append(j)
                    break

        print(f"Found {len(self.foot_geom_ids)} foot geoms")

        if self.debug:
            print(f"Initial robot height: {self.model.stat.center[2]:.3f}")
            print(f"Camera name: {self.camera_name}")
            # Print initial orientation using current data
            print(f"Initial orientation (quat): {self.data.qpos[3:7]}")

        # Get actuator limits from the model
        self.actuator_limits = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if self.model.jnt_limited[joint_id]:
                self.actuator_limits.append(
                    (
                        self.model.jnt_range[joint_id, 0],
                        self.model.jnt_range[joint_id, 1],
                    )
                )
            else:
                self.actuator_limits.append((-np.pi, np.pi))  # Default limits

        # Simplified action space: just target positions for each actuator
        pos_lows = np.array([lim[0] for lim in self.actuator_limits])
        pos_highs = np.array([lim[1] for lim in self.actuator_limits])
        self.action_space = spaces.Box(
            low=pos_lows,
            high=pos_highs,
            dtype=np.float32,
        )

        # Define side groups for lateral stability
        self.side_groups = {
            "left": [0, 1, 2, 3],  # Legs 1,2,3,4
            "right": [4, 5, 6, 7],  # Legs 5,6,7,8
        }

        # Debug: print joint limits for each leg
        if self.debug:
            print("Joint limits for each leg:")
            for i in range(8):
                leg_start = i * 3
                print(
                    f"Leg {i+1}: Hip {self.actuator_limits[leg_start]}, Femur {self.actuator_limits[leg_start+1]}, Tibia {self.actuator_limits[leg_start+2]}"
                )

    def _get_obs_dim(self, xml_file):
        """Calculate observation dimension based on the model."""
        # The actual observation includes:
        # - Joint positions (full qpos) = 31
        # - Joint velocities (full qvel) = 30
        # - IMU data (orientation 4 + angular velocity 3 + linear acceleration 3) = 10
        # - Body position (3) + Body orientation as Euler angles (3) + Height (1) = 7
        # - Contact forces (8) = 8
        # - Gait phase (1) = 1
        # Total: 31 + 30 + 10 + 7 + 8 + 1 = 87
        return 87

    def step(self, action):
        """Execute one timestep of the environment dynamics (position control only)."""
        # Simplified control: just position targets
        pos = np.clip(
            action,
            [lim[0] for lim in self.actuator_limits],
            [lim[1] for lim in self.actuator_limits],
        )

        # Get current joint states
        current_positions = self.data.qpos[7:31]
        current_velocities = self.data.qvel[6:30]

        # PD control law: tau = Kp*(q_des - q) + Kd*(-qd)
        torques = (
            self.position_gain * (pos - current_positions)
            - self.velocity_gain * current_velocities
        )
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        # Debug: print action statistics occasionally
        if self.debug and self.steps_taken % 100 == 0:
            print(
                f"Step {self.steps_taken}: pos range [{pos.min():.3f}, {pos.max():.3f}]"
            )

        self.do_simulation(torques, self.frame_skip)

        # Update gait phase
        self.gait_phase += self.dt * self.frame_skip * self.gait_frequency * 2 * np.pi
        self.gait_phase = self.gait_phase % (2 * np.pi)

        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()
        self.steps_taken += 1
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        """Reset the robot to initial position."""
        # Set initial joint positions (standing pose)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add very small random perturbations for robustness
        qpos[7:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq - 7)
        qvel[6:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv - 6)

        self.set_state(qpos, qvel)

        # Reset tracking variables
        self.previous_x_position = self.data.qpos[0]
        self.steps_taken = 0
        self.gait_phase = 0.0
        self.contact_history = []
        self.previous_hip_positions = {}
        self.previous_foot_positions = {}
        self.step_lengths = []
        self.previous_joint_velocities = None
        self.joint_acceleration_history = []

        # Set target height based on initial standing position
        if not hasattr(self, "target_height"):
            self.target_height = self.data.qpos[2]
            print(f"Set target height to: {self.target_height:.3f}")

        # Debug: print initial state
        if self.debug:
            print(f"Reset - height: {self.data.qpos[2]:.3f}")
            orientation = self.data.xquat[self.torso_id]
            print(
                f"Reset - orientation (x,y,z,w): ({orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}, {orientation[3]:.3f})"
            )

        return self._get_obs()

    def _get_obs(self):
        """Get current observation including IMU data and proprioception."""
        # Robot state
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Orientation (quaternion)
        orientation = self.data.xquat[self.torso_id]

        # Angular velocity
        angular_vel = self.data.cvel[self.torso_id][3:6]

        # Linear acceleration (approximate using finite differences)
        linear_acc = self.data.qacc[:3] if hasattr(self.data, "qacc") else np.zeros(3)

        # Body position and orientation
        body_pos = self.data.xpos[self.torso_id]
        body_mat = self.data.xmat[self.torso_id].reshape(3, 3)
        body_euler = self._mat2euler(body_mat)

        # Contact forces for each foot
        contact_forces = np.zeros(8)
        for idx, foot_geom_id in enumerate(self.foot_geom_ids):
            force_magnitude = 0.0
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # Get contact force magnitude
                    # Try to get actual force from MuJoCo contact data
                    if hasattr(contact, "efc_force") and j < len(contact.efc_force):
                        force = np.sqrt(np.sum(contact.efc_force[j] ** 2))
                    else:
                        force = 1.0  # Binary contact if force not available
                    force_magnitude = max(force_magnitude, force)
            contact_forces[idx] = force_magnitude

        # Combine all observations
        obs = np.concatenate(
            [
                qpos,  # Joint positions
                qvel,  # Joint velocities
                orientation,  # IMU orientation (4)
                angular_vel,  # IMU angular velocity (3)
                linear_acc,  # IMU linear acceleration (3)
                body_pos,  # Body position (3)
                body_euler,  # Body orientation as Euler angles (3)
                [self.data.qpos[2]],  # Height (1)
                contact_forces,  # Foot contact forces (8)
                [np.sin(self.gait_phase)],  # Gait phase (1)
            ]
        )

        return obs

    def _compute_reward(self):
        """Compute reward based on forward progress, stability, and efficiency."""
        # Get robot's current orientation
        orientation = self.data.xquat[self.torso_id]
        body_mat = self.data.xmat[self.torso_id].reshape(3, 3)

        # Get forward direction vector (robot's local X-axis)
        forward_direction = body_mat[:, 0]  # First column is forward direction

        # Calculate velocity in robot's forward direction
        velocity = self.data.qvel[:3]  # Linear velocity
        forward_velocity = np.dot(velocity, forward_direction)

        # Forward reward based on robot's orientation
        forward_reward = forward_velocity
        forward_reward = np.clip(forward_reward, -1.0, 2.0)

        # Height reward (encourage maintaining height) - critical for stability
        current_height = self.data.qpos[2]
        if not hasattr(self, "target_height"):
            self.target_height = current_height

        # Strong height reward with tight tolerance
        height_error = abs(current_height - self.target_height)
        height_reward = 5.0 * np.exp(-20 * height_error**2)

        # Orientation reward (stay upright)
        target_orientation = np.array([0.707, 0.707, 0.0, 0.0])
        orientation_error = np.sum((orientation - target_orientation) ** 2)
        upright_reward = 3.0 * np.exp(-5 * orientation_error)

        # Body stability (minimize angular velocity)
        angular_vel = self.data.cvel[self.torso_id][3:6]
        stability_reward = 1.0 * np.exp(-0.5 * np.sum(angular_vel**2))

        # Lateral drift penalty (stay on course)
        lateral_penalty = -0.5 * abs(self.data.qpos[1])

        # Energy efficiency penalty - reduced to allow movement
        energy_penalty = -0.00005 * np.sum(np.square(self.data.ctrl))

        # Foot contact and gait pattern reward
        contact_pattern_reward, gait_quality = self._compute_gait_reward()

        # Weight distribution reward (new!)
        weight_distribution_reward = self._compute_weight_distribution_reward()

        # Stride length reward - encourage larger steps
        stride_reward = self._compute_stride_reward()

        # Joint limit penalty
        joint_positions = self.data.qpos[7:31]
        joint_limit_penalty = 0
        for i, (pos, (low, high)) in enumerate(
            zip(joint_positions, self.actuator_limits)
        ):
            margin = 0.1 * (high - low)  # 10% margin
            if pos < low + margin or pos > high - margin:
                joint_limit_penalty -= 0.1

        # Joint smoothness penalty (penalize excessive acceleration/jerk)
        smoothness_penalty = self._compute_smoothness_penalty()

        # Leg coordination reward (encourage coordinated hip-femur-tibia movement)
        leg_coordination_reward = self._compute_leg_coordination_reward()

        # Survival bonus
        survival_bonus = 1.0

        # Total reward with emphasis on stability
        reward = (
            1.0 * forward_reward  # Moderate forward reward
            + height_reward  # Critical height maintenance
            + upright_reward  # Critical orientation
            + stability_reward  # Body stability
            + 2.0 * contact_pattern_reward  # Gait pattern
            + 0.5 * gait_quality  # Gait coordination
            + 1.5 * weight_distribution_reward  # Weight distribution
            + 1.0 * stride_reward  # Stride length
            + 1.0 * leg_coordination_reward  # Leg coordination
            + survival_bonus
            - energy_penalty
            - lateral_penalty
            + joint_limit_penalty
            + smoothness_penalty  # Joint smoothness (negative penalty)
        )

        return reward

    def _compute_stride_reward(self):
        """Reward for actual hip movement that contributes to forward progress."""
        joint_positions = self.data.qpos[7:31]
        joint_velocities = self.data.qvel[6:30]  # Joint velocities

        # Hip joints are at indices 0, 3, 6, 9, 12, 15, 18, 21 (every 3rd joint)
        hip_indices = [0, 3, 6, 9, 12, 15, 18, 21]

        # Get robot's forward direction
        body_mat = self.data.xmat[self.torso_id].reshape(3, 3)
        forward_direction = body_mat[:, 0]  # Robot's local X-axis

        # Track actual step lengths
        current_step_length = self._track_step_lengths()

        total_stride_reward = 0.0
        total_movement_reward = 0.0

        for hip_idx in hip_indices:
            if hip_idx < len(joint_positions):
                hip_position = joint_positions[hip_idx]
                hip_velocity = joint_velocities[hip_idx]
                hip_limits = self.actuator_limits[hip_idx]

                # Calculate how much of the range is being used
                range_size = hip_limits[1] - hip_limits[0]
                position_in_range = (hip_position - hip_limits[0]) / range_size

                # 1. Position reward - encourage using more of the range (but not extremes)
                if 0.2 <= position_in_range <= 0.8:
                    # Reward for using more of the middle range
                    position_reward = 2.0 * (1.0 - abs(position_in_range - 0.5))
                elif 0.1 <= position_in_range <= 0.9:
                    position_reward = 1.0 * (1.0 - abs(position_in_range - 0.5))
                else:
                    position_reward = -0.5  # Penalty for extremes

                # 2. Movement reward - encourage larger hip movements
                # Track hip position changes over time
                if not hasattr(self, "previous_hip_positions"):
                    self.previous_hip_positions = {i: hip_position for i in hip_indices}

                hip_movement = abs(
                    hip_position
                    - self.previous_hip_positions.get(hip_idx, hip_position)
                )
                self.previous_hip_positions[hip_idx] = hip_position

                # Reward for larger hip movements (encourages bigger steps)
                # Normalize by joint range to get relative movement
                relative_movement = hip_movement / range_size

                if relative_movement > 0.1:  # Significant movement (>10% of range)
                    movement_reward = (
                        3.0 * relative_movement
                    )  # Strong reward for large movements
                elif relative_movement > 0.05:  # Moderate movement
                    movement_reward = 1.0 * relative_movement
                else:
                    movement_reward = -0.5  # Penalty for very small movements

                # 3. Velocity reward - encourage purposeful hip movement
                velocity_magnitude = abs(hip_velocity)
                if 0.3 <= velocity_magnitude <= 2.5:  # Good speed range
                    velocity_reward = 1.0
                elif 0.1 <= velocity_magnitude <= 3.0:  # Acceptable range
                    velocity_reward = 0.5
                else:
                    velocity_reward = -0.3  # Penalty for too slow or too fast

                # 4. Directional movement reward - encourage hip movement that contributes to forward motion
                robot_forward_velocity = np.dot(self.data.qvel[:3], forward_direction)

                # If robot is moving forward and hip is moving, that's good
                if robot_forward_velocity > 0.1 and abs(hip_velocity) > 0.2:
                    directional_reward = 1.0
                elif robot_forward_velocity > 0.05 and abs(hip_velocity) > 0.1:
                    directional_reward = 0.5
                else:
                    directional_reward = 0.0

                # Combine all rewards for this hip
                hip_total_reward = (
                    0.3 * position_reward
                    + 0.4 * movement_reward  # Emphasize actual movement
                    + 0.2 * velocity_reward
                    + 0.1 * directional_reward
                )

                total_stride_reward += hip_total_reward
                total_movement_reward += movement_reward

        # Add step length reward - encourage larger actual steps
        step_length_reward = 0.0
        if current_step_length > 0.05:  # Significant step length
            step_length_reward = (
                5.0 * current_step_length
            )  # Strong reward for large steps
        elif current_step_length > 0.02:  # Moderate step length
            step_length_reward = 2.0 * current_step_length
        else:
            step_length_reward = -1.0  # Penalty for very small steps

        # Add step length reward to total
        total_stride_reward += step_length_reward

        # Average rewards across all hips
        avg_stride_reward = total_stride_reward / len(hip_indices)
        avg_movement_reward = total_movement_reward / len(hip_indices)

        # Debug output
        if self.debug and self.steps_taken % 200 == 0:
            print(
                f"Stride reward: total={avg_stride_reward:.3f}, movement={avg_movement_reward:.3f}, step_length={current_step_length:.4f}"
            )
            # Print hip movements for first few legs
            for i, hip_idx in enumerate(hip_indices[:3]):  # Show first 3 legs
                if hip_idx < len(joint_positions):
                    pos = joint_positions[hip_idx]
                    vel = joint_velocities[hip_idx]
                    limits = self.actuator_limits[hip_idx]
                    range_used = (pos - limits[0]) / (limits[1] - limits[0])
                    movement = abs(pos - self.previous_hip_positions.get(hip_idx, pos))
                    relative_movement = movement / (limits[1] - limits[0])
                    print(
                        f"  Leg{i+1} hip: pos={pos:.3f}, vel={vel:.3f}, range_used={range_used:.2f}, movement={relative_movement:.3f}"
                    )

        return avg_stride_reward

    def _compute_gait_reward(self):
        """Reward for maintaining a coordinated gait pattern."""
        # Get current foot contacts
        feet_in_contact = []
        for idx, foot_geom_id in enumerate(self.foot_geom_ids):
            in_contact = False
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    in_contact = True
                    break
            feet_in_contact.append(in_contact)

        # Update contact history
        self.contact_history.append(feet_in_contact)
        if len(self.contact_history) > self.max_contact_history:
            self.contact_history.pop(0)

        # Count feet in contact
        num_feet_in_contact = sum(feet_in_contact)

        # Basic contact reward (4-6 feet ideal for stability)
        if 4 <= num_feet_in_contact <= 6:
            contact_reward = 1.0
        elif num_feet_in_contact == 3 or num_feet_in_contact == 7:
            contact_reward = 0.5
        else:
            contact_reward = 0.0

        # Gait coordination reward based on tripod pattern
        gait_quality = 0.0
        if len(self.contact_history) >= 10:
            # Check if legs in each group move together
            group_a_sync = 0
            group_b_sync = 0

            for contact_state in self.contact_history[-10:]:
                # Group A coordination
                group_a_contacts = [
                    contact_state[i] for i in self.leg_groups["group_a"]
                ]
                if all(group_a_contacts) or not any(group_a_contacts):
                    group_a_sync += 0.1

                # Group B coordination
                group_b_contacts = [
                    contact_state[i] for i in self.leg_groups["group_b"]
                ]
                if all(group_b_contacts) or not any(group_b_contacts):
                    group_b_sync += 0.1

            # Reward alternating pattern between groups
            if len(self.contact_history) >= 2:
                current_a = [feet_in_contact[i] for i in self.leg_groups["group_a"]]
                current_b = [feet_in_contact[i] for i in self.leg_groups["group_b"]]
                prev_a = [
                    self.contact_history[-2][i] for i in self.leg_groups["group_a"]
                ]
                prev_b = [
                    self.contact_history[-2][i] for i in self.leg_groups["group_b"]
                ]

                # Check for alternation
                if (sum(current_a) > sum(prev_a) and sum(current_b) < sum(prev_b)) or (
                    sum(current_a) < sum(prev_a) and sum(current_b) > sum(prev_b)
                ):
                    gait_quality += 0.5

            gait_quality += (group_a_sync + group_b_sync) * 0.5

        return contact_reward, gait_quality

    def _compute_weight_distribution_reward(self):
        """Reward for even weight distribution across contacting feet."""
        # Get contact forces for each foot
        contact_forces = np.zeros(8)
        feet_in_contact = []

        for idx, foot_geom_id in enumerate(self.foot_geom_ids):
            force_magnitude = 0.0
            in_contact = False

            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # Calculate contact force magnitude
                    # MuJoCo stores contact forces in the contact object
                    force = (
                        np.sqrt(np.sum(contact.efc_force[j] ** 2))
                        if hasattr(contact, "efc_force")
                        else 1.0
                    )
                    force_magnitude = max(force_magnitude, force)
                    in_contact = True

            contact_forces[idx] = force_magnitude
            feet_in_contact.append(in_contact)

        # Only consider feet that are actually in contact
        contacting_forces = [contact_forces[i] for i in range(8) if feet_in_contact[i]]

        if len(contacting_forces) < 2:
            return 0.0  # Need at least 2 feet in contact

        # Calculate weight distribution metrics
        total_force = sum(contacting_forces)
        mean_force = total_force / len(contacting_forces)

        # Variance in force distribution (lower is better)
        force_variance = (
            np.var(contacting_forces) if len(contacting_forces) > 1 else 0.0
        )

        # Coefficient of variation (standardized measure of dispersion)
        cv = np.sqrt(force_variance) / mean_force if mean_force > 0 else 1.0

        # Reward for even distribution (lower CV is better)
        distribution_reward = np.exp(-3.0 * cv)

        # Lateral balance reward (left vs right side)
        left_forces = [
            contact_forces[i] for i in self.side_groups["left"] if feet_in_contact[i]
        ]
        right_forces = [
            contact_forces[i] for i in self.side_groups["right"] if feet_in_contact[i]
        ]

        left_total = sum(left_forces) if left_forces else 0.0
        right_total = sum(right_forces) if right_forces else 0.0

        # Balance between left and right sides
        if left_total + right_total > 0:
            balance_ratio = min(left_total, right_total) / max(left_total, right_total)
            lateral_balance_reward = balance_ratio
        else:
            lateral_balance_reward = 0.0

        # Combined weight distribution reward
        total_weight_reward = 0.7 * distribution_reward + 0.3 * lateral_balance_reward

        # Debug output
        if self.debug and self.steps_taken % 200 == 0:
            print(
                f"Weight distribution: CV={cv:.3f}, balance={lateral_balance_reward:.3f}, reward={total_weight_reward:.3f}"
            )
            print(f"Contact forces: {contact_forces}")

        return total_weight_reward

    def _is_terminated(self):
        """Check if episode should terminate (robot fell or flipped)."""
        # Check height - robot normally stands at 0.134
        height = self.data.qpos[2]
        if height < 0.05:  # Slightly more lenient
            if self.debug:
                print(f"Terminated: Low height {height:.3f}")
            return True

        # Check orientation (if robot flipped)
        orientation = self.data.xquat[self.torso_id]
        # Check if orientation has deviated too much from upright
        target_orientation = np.array([0.707, 0.707, 0.0, 0.0])
        orientation_error = np.sum((orientation - target_orientation) ** 2)
        if orientation_error > 1.0:  # Significant deviation
            if self.debug:
                print(
                    f"Terminated: Bad orientation error={orientation_error:.3f}, quat=({orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}, {orientation[3]:.3f})"
                )
            return True

        # Check if robot has gone too far off course (sideways)
        y_position = abs(self.data.qpos[1])
        if y_position > 5.0:
            if self.debug:
                print(f"Terminated: Lateral drift {y_position:.3f}")
            return True

        # Check for NaN values (simulation instability)
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print("Warning: NaN detected in simulation, terminating episode")
            return True

        return False

    def _get_info(self):
        """Return additional information about the episode."""
        # Calculate weight distribution metrics for info
        contact_forces = np.zeros(8)
        feet_in_contact = []

        for idx, foot_geom_id in enumerate(self.foot_geom_ids):
            force_magnitude = 0.0
            in_contact = False

            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    if hasattr(contact, "efc_force") and j < len(contact.efc_force):
                        force = np.sqrt(np.sum(contact.efc_force[j] ** 2))
                    else:
                        force = 1.0
                    force_magnitude = max(force_magnitude, force)
                    in_contact = True

            contact_forces[idx] = force_magnitude
            feet_in_contact.append(in_contact)

        contacting_forces = [contact_forces[i] for i in range(8) if feet_in_contact[i]]
        num_feet_in_contact = len(contacting_forces)

        # Weight distribution metrics
        weight_distribution_cv = 0.0
        lateral_balance = 0.0

        if num_feet_in_contact >= 2:
            total_force = sum(contacting_forces)
            mean_force = total_force / num_feet_in_contact
            force_variance = np.var(contacting_forces)
            weight_distribution_cv = (
                np.sqrt(force_variance) / mean_force if mean_force > 0 else 1.0
            )

            # Lateral balance
            left_forces = [
                contact_forces[i]
                for i in self.side_groups["left"]
                if feet_in_contact[i]
            ]
            right_forces = [
                contact_forces[i]
                for i in self.side_groups["right"]
                if feet_in_contact[i]
            ]
            left_total = sum(left_forces) if left_forces else 0.0
            right_total = sum(right_forces) if right_forces else 0.0

            if left_total + right_total > 0:
                lateral_balance = min(left_total, right_total) / max(
                    left_total, right_total
                )

        return {
            "x_position": self.data.qpos[0],
            "height": self.data.qpos[2],
            "forward_speed": self.data.qvel[0],
            "energy_used": np.sum(np.square(self.data.ctrl)),
            "steps": self.steps_taken,
            "num_feet_in_contact": num_feet_in_contact,
            "weight_distribution_cv": weight_distribution_cv,
            "lateral_balance": lateral_balance,
            "total_contact_force": sum(contact_forces),
        }

    def _mat2euler(self, mat):
        """Convert rotation matrix to Euler angles."""
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(mat[2, 1], mat[2, 2])
            y = np.arctan2(-mat[2, 0], sy)
            z = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            x = np.arctan2(-mat[1, 2], mat[1, 1])
            y = np.arctan2(-mat[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def _track_step_lengths(self):
        """Track actual step lengths by monitoring foot positions."""
        # Get foot positions in world coordinates
        foot_positions = []
        for foot_geom_id in self.foot_geom_ids:
            # Get the body ID from the geom ID
            foot_body_id = self.model.geom_bodyid[foot_geom_id]
            foot_pos = self.data.xpos[foot_body_id]
            foot_positions.append(foot_pos.copy())

        # Track foot movement and step lengths
        if (
            not hasattr(self, "previous_foot_positions")
            or not self.previous_foot_positions
        ):
            self.previous_foot_positions = {
                i: pos.copy() for i, pos in enumerate(foot_positions)
            }
            return 0.0

        total_step_length = 0.0
        num_feet_moving = 0

        for i, (current_pos, prev_pos) in enumerate(
            zip(foot_positions, self.previous_foot_positions.values())
        ):
            # Calculate foot movement in robot's forward direction
            body_mat = self.data.xmat[self.torso_id].reshape(3, 3)
            forward_direction = body_mat[:, 0]

            # Project foot movement onto forward direction
            foot_movement = current_pos - prev_pos
            forward_movement = np.dot(foot_movement, forward_direction)

            # Only count positive forward movement (actual steps)
            if forward_movement > 0.01:  # Significant forward movement
                total_step_length += forward_movement
                num_feet_moving += 1

            # Update previous position
            self.previous_foot_positions[i] = current_pos.copy()

        # Calculate average step length
        avg_step_length = total_step_length / max(num_feet_moving, 1)

        # Store step length history
        self.step_lengths.append(avg_step_length)
        if len(self.step_lengths) > self.max_step_history:
            self.step_lengths.pop(0)

        return avg_step_length

    def _compute_smoothness_penalty(self):
        """Penalize excessive joint acceleration/jerk to encourage smooth movement."""
        current_joint_velocities = self.data.qvel[6:30]  # Joint velocities

        if self.previous_joint_velocities is None:
            self.previous_joint_velocities = current_joint_velocities.copy()
            return 0.0

        # Calculate joint accelerations (change in velocity)
        joint_accelerations = (
            current_joint_velocities - self.previous_joint_velocities
        ) / (self.dt * self.frame_skip)

        # Store acceleration history
        self.joint_acceleration_history.append(joint_accelerations.copy())
        if len(self.joint_acceleration_history) > self.max_acceleration_history:
            self.joint_acceleration_history.pop(0)

        # Update previous velocities
        self.previous_joint_velocities = current_joint_velocities.copy()

        # Calculate smoothness penalty
        # 1. Acceleration magnitude penalty (penalize large accelerations)
        acc_magnitude_penalty = -0.1 * np.sum(np.square(joint_accelerations))

        # 2. Direction change penalty (penalize rapid velocity direction changes)
        direction_change_penalty = 0.0
        if len(self.joint_acceleration_history) >= 2:
            prev_acc = self.joint_acceleration_history[-2]
            curr_acc = self.joint_acceleration_history[-1]

            # Penalize when acceleration changes sign rapidly (jerk)
            for i in range(len(prev_acc)):
                if (
                    abs(prev_acc[i]) > 0.1 and abs(curr_acc[i]) > 0.1
                ):  # Only if significant acceleration
                    if np.sign(prev_acc[i]) != np.sign(curr_acc[i]):  # Direction change
                        direction_change_penalty -= 0.05

        # 3. Velocity consistency penalty (penalize erratic velocity changes)
        velocity_consistency_penalty = 0.0
        if len(self.joint_acceleration_history) >= 3:
            # Check for consistent acceleration patterns
            recent_accs = np.array(self.joint_acceleration_history[-3:])
            acc_variance = np.var(recent_accs, axis=0)
            velocity_consistency_penalty = -0.02 * np.sum(acc_variance)

        total_smoothness_penalty = (
            acc_magnitude_penalty
            + direction_change_penalty
            + velocity_consistency_penalty
        )

        # Debug output
        if self.debug and self.steps_taken % 200 == 0:
            acc_rms = np.sqrt(np.mean(np.square(joint_accelerations)))
            print(
                f"Smoothness: acc_rms={acc_rms:.3f}, dir_changes={direction_change_penalty:.3f}, penalty={total_smoothness_penalty:.3f}"
            )

        return total_smoothness_penalty

    def _compute_leg_coordination_reward(self):
        """Reward coordinated movement between hip, femur, and tibia joints within each leg."""
        joint_positions = self.data.qpos[7:31]
        joint_velocities = self.data.qvel[6:30]

        total_coordination_reward = 0.0

        # Analyze each leg (8 legs, 3 joints each)
        for leg_idx in range(8):
            leg_start = leg_idx * 3

            # Get joint indices for this leg
            hip_idx = leg_start + 0
            femur_idx = leg_start + 1
            tibia_idx = leg_start + 2

            if tibia_idx >= len(joint_positions):
                continue

            # Get joint states
            hip_pos = joint_positions[hip_idx]
            femur_pos = joint_positions[femur_idx]
            tibia_pos = joint_positions[tibia_idx]

            hip_vel = joint_velocities[hip_idx]
            femur_vel = joint_velocities[femur_idx]
            tibia_vel = joint_velocities[tibia_idx]

            # 1. Velocity coordination reward
            # Encourage joints to move in coordinated patterns
            velocity_magnitudes = np.array(
                [abs(hip_vel), abs(femur_vel), abs(tibia_vel)]
            )

            # Reward when all joints are active (not just hip)
            if np.all(velocity_magnitudes > 0.1):  # All joints moving significantly
                velocity_coordination = 1.0
            elif np.sum(velocity_magnitudes > 0.1) >= 2:  # At least 2 joints moving
                velocity_coordination = 0.5
            else:
                velocity_coordination = -0.2  # Penalty for only one joint moving

            # 2. Phase relationship reward
            # Encourage proper phase relationships between joints
            # Hip leads, femur follows, tibia completes the motion
            phase_reward = 0.0

            # Check if joints are moving in coordinated directions
            if abs(hip_vel) > 0.1 and abs(femur_vel) > 0.1:
                # Reward when hip and femur move in coordinated patterns
                if np.sign(hip_vel) == np.sign(femur_vel):
                    phase_reward += (
                        0.3  # Same direction can be good for some gait phases
                    )
                else:
                    phase_reward += 0.5  # Opposite directions often better for walking

            if abs(femur_vel) > 0.1 and abs(tibia_vel) > 0.1:
                # Femur and tibia coordination
                if abs(np.sign(femur_vel) - np.sign(tibia_vel)) < 0.1:
                    phase_reward += 0.3

            # 3. Range utilization reward
            # Encourage using more of each joint's range, not just hip
            range_utilization = 0.0
            for joint_idx, joint_pos in enumerate([hip_pos, femur_pos, tibia_pos]):
                actual_joint_idx = leg_start + joint_idx
                if actual_joint_idx < len(self.actuator_limits):
                    joint_limits = self.actuator_limits[actual_joint_idx]
                    range_size = joint_limits[1] - joint_limits[0]
                    position_in_range = (joint_pos - joint_limits[0]) / range_size

                    # Reward for using middle portions of the range
                    if 0.2 <= position_in_range <= 0.8:
                        range_utilization += 0.3
                    elif 0.1 <= position_in_range <= 0.9:
                        range_utilization += 0.1
                    else:
                        range_utilization -= 0.1  # Penalty for extremes

            # 4. Movement amplitude balance
            # Encourage similar movement amplitudes across joints
            amplitude_balance = 0.0
            if np.all(velocity_magnitudes > 0.05):
                # Calculate coefficient of variation for velocity magnitudes
                mean_vel = np.mean(velocity_magnitudes)
                vel_cv = np.std(velocity_magnitudes) / mean_vel if mean_vel > 0 else 1.0
                amplitude_balance = 0.5 * np.exp(
                    -2.0 * vel_cv
                )  # Reward balanced movement

            # Combine rewards for this leg
            leg_coordination = (
                0.4 * velocity_coordination
                + 0.3 * phase_reward
                + 0.2 * range_utilization
                + 0.1 * amplitude_balance
            )

            total_coordination_reward += leg_coordination

        # Average across all legs
        avg_coordination_reward = total_coordination_reward / 8

        # Debug output
        if self.debug and self.steps_taken % 200 == 0:
            print(f"Leg coordination reward: {avg_coordination_reward:.3f}")
            # Show details for first few legs
            for leg_idx in range(min(3, 8)):
                leg_start = leg_idx * 3
                if leg_start + 2 < len(joint_velocities):
                    hip_vel = joint_velocities[leg_start]
                    femur_vel = joint_velocities[leg_start + 1]
                    tibia_vel = joint_velocities[leg_start + 2]
                    print(
                        f"  Leg{leg_idx+1}: hip_vel={hip_vel:.3f}, femur_vel={femur_vel:.3f}, tibia_vel={tibia_vel:.3f}"
                    )

        return avg_coordination_reward


def make_env(xml_file, debug=False):
    """Create a wrapped environment for parallel training."""

    def _init():
        env = SpiderRobotEnv(
            xml_file, render_mode="rgb_array", camera_name="bodycam", debug=debug
        )
        env = Monitor(env)
        return env

    return _init
