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
        self.previous_x_position = 0
        self.steps_taken = 0
        self.fall_count = 0
        self.camera_name = camera_name  # Store camera name for rendering
        self.debug = debug  # Debug mode

        # Gait phase tracking
        self.gait_phase = 0.0
        self.gait_frequency = 0.5  # Hz

        # Contact history for gait reward
        self.contact_history = []
        self.max_contact_history = 50

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

        # Define leg groups for gait patterns
        # Tripod gait: two groups that alternate
        self.leg_groups = {
            "group_a": [0, 2, 5, 7],  # Legs 1, 3, 6, 8
            "group_b": [1, 3, 4, 6],  # Legs 2, 4, 5, 7
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
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # Get contact force magnitude - use a simpler approach
                    # The contact force is available in the contact object
                    contact_forces[idx] = 1.0  # Binary contact

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
        # Forward progress reward - moderate
        current_x = self.data.qpos[0]
        forward_reward = (current_x - self.previous_x_position) / self.dt
        self.previous_x_position = current_x

        # Clip forward reward to prevent exploitation
        forward_reward = np.clip(forward_reward, -1.0, 2.0)

        # Height reward (encourage maintaining height) - critical for stability
        current_height = self.data.qpos[2]
        if not hasattr(self, "target_height"):
            self.target_height = current_height

        # Strong height reward with tight tolerance
        height_error = abs(current_height - self.target_height)
        height_reward = 5.0 * np.exp(-20 * height_error**2)

        # Orientation reward (stay upright) - critical
        orientation = self.data.xquat[self.torso_id]
        # The robot's upright orientation has specific quaternion values
        # Based on the initial orientation (0.707, 0.707, 0, 0)
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

        # Joint limit penalty
        joint_positions = self.data.qpos[7:31]
        joint_limit_penalty = 0
        for i, (pos, (low, high)) in enumerate(
            zip(joint_positions, self.actuator_limits)
        ):
            margin = 0.1 * (high - low)  # 10% margin
            if pos < low + margin or pos > high - margin:
                joint_limit_penalty -= 0.1

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
            + survival_bonus
            - energy_penalty
            - lateral_penalty
            + joint_limit_penalty
        )

        return reward

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
        return {
            "x_position": self.data.qpos[0],
            "height": self.data.qpos[2],
            "forward_speed": self.data.qvel[0],
            "energy_used": np.sum(np.square(self.data.ctrl)),
            "steps": self.steps_taken,
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


def make_env(xml_file, debug=False):
    """Create a wrapped environment for parallel training."""

    def _init():
        env = SpiderRobotEnv(
            xml_file, render_mode="rgb_array", camera_name="bodycam", debug=debug
        )
        env = Monitor(env)
        return env

    return _init
