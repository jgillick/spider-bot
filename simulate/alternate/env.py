"""
Spider Robot Reinforcement Learning Environment and Training
Uses Gymnasium and MuJoCo for physics simulation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecVideoRecorder,
)
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.evaluation import evaluate_policy


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

        # Control parameters
        self.max_torque = 15.0
        self.position_gain = 20.0
        self.velocity_gain = 1.0

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

        # Action space: target positions for each actuator
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in self.actuator_limits]),
            high=np.array([lim[1] for lim in self.actuator_limits]),
            dtype=np.float32,
        )

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
        # - Joint positions (24) + Joint velocities (24) = 48
        # - IMU data (orientation 4 + angular velocity 3 + linear acceleration 3) = 10
        # - Body position (3) + Body orientation as Euler angles (3) + Height (1) = 7
        # - Contact forces (8) = 8
        # Total: 48 + 10 + 7 + 8 = 73
        # But the actual observation is 86, so let's use that
        return 86

    def step(self, action):
        """Execute one timestep of the environment dynamics."""
        if isinstance(self.action_space, gym.spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = action  # or handle other space types if needed

        # Debug: print action statistics occasionally
        if self.debug and self.steps_taken % 100 == 0:
            print(
                f"Step {self.steps_taken}: action range [{action.min():.3f}, {action.max():.3f}], mean {action.mean():.3f}"
            )
            # Print individual leg actions
            for i in range(8):
                leg_start = i * 3
                print(
                    f"  Leg {i+1}: Hip {action[leg_start]:.3f}, Femur {action[leg_start+1]:.3f}, Tibia {action[leg_start+2]:.3f}"
                )

        # Convert position targets to torques using PD control
        current_positions = self.data.qpos[7:31]  # Skip root joint (first 7 values)
        current_velocities = self.data.qvel[6:30]  # Skip root joint (first 6 values)

        position_error = action - current_positions
        torques = (
            self.position_gain * position_error
            - self.velocity_gain * current_velocities
        )

        # Clip torques to maximum
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        # Debug: print torque statistics occasionally
        if self.debug and self.steps_taken % 100 == 0:
            print(
                f"Step {self.steps_taken}: torque range [{torques.min():.3f}, {torques.max():.3f}], mean {torques.mean():.3f}"
            )
            # Print individual leg torques
            for i in range(8):
                leg_start = i * 3
                print(
                    f"  Leg {i+1}: Hip {torques[leg_start]:.3f}, Femur {torques[leg_start+1]:.3f}, Tibia {torques[leg_start+2]:.3f}"
                )

        # Apply torques
        self.do_simulation(torques, self.frame_skip)

        # Calculate reward and check termination
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
            # Print initial joint positions for each leg
            print("Initial joint positions:")
            for i in range(8):
                leg_start = 7 + i * 3
                print(
                    f"Leg {i+1}: Hip {qpos[leg_start]:.3f}, Femur {qpos[leg_start+1]:.3f}, Tibia {qpos[leg_start+2]:.3f}"
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
                    contact_forces[idx] = (
                        np.linalg.norm(contact.efc_pos[:3])
                        if hasattr(contact, "efc_pos")
                        else 0.0
                    )

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
                [self.data.qpos[2]],  # Height
                contact_forces,  # Foot contact forces (8)
            ]
        )

        return obs

    def _compute_reward(self):
        """Compute reward based on forward progress, stability, and efficiency."""
        # Forward progress reward - make this much stronger
        current_x = self.data.qpos[0]
        forward_reward = (current_x - self.previous_x_position) / self.dt
        self.previous_x_position = current_x

        # Clip forward reward to prevent exploitation but allow more movement
        forward_reward = np.clip(
            forward_reward, -1.0, 5.0
        )  # Allow more positive reward

        # Height reward (encourage maintaining height) - much stronger
        current_height = self.data.qpos[2]
        if not hasattr(self, "target_height"):
            self.target_height = current_height

        # Strong height reward to prevent collapsing
        height_error = abs(current_height - self.target_height)
        height_reward = 10.0 * np.exp(
            -5 * height_error**2
        )  # Much stronger height reward

        # Orientation reward (stay close to initial orientation) - stronger
        orientation = self.data.xquat[self.torso_id]
        target_orientation = np.array([0.707, 0.707, 0.0, 0.0])
        orientation_diff = np.sum((orientation - target_orientation) ** 2)
        upright_reward = 3.0 * np.exp(
            -3 * orientation_diff
        )  # Stronger orientation reward

        # Lateral drift penalty (stay on course) - moderate penalty
        lateral_penalty = 0.2 * abs(self.data.qpos[1])  # Moderate penalty

        # Energy efficiency penalty - reduce to allow more movement
        energy_penalty = 0.00001 * np.sum(
            np.square(self.data.ctrl)
        )  # Very small energy penalty

        # Foot contact reward (encourage alternating gait)
        contact_pattern_reward = self._compute_gait_reward()

        # Smoothness reward (penalize jerky movements) - reduce penalty
        smoothness_penalty = 0.0001 * np.sum(
            np.square(self.data.qvel[6:])
        )  # Small smoothness penalty

        # Survival bonus (encourage staying alive)
        survival_bonus = 0.5

        # Total reward - prioritize stability and movement
        reward = (
            2.0 * forward_reward  # Strong forward reward
            + 3.0 * height_reward  # Very strong height reward
            + upright_reward  # Strong orientation reward
            + 2.0 * contact_pattern_reward  # Strong contact reward
            + survival_bonus
            - energy_penalty
            - smoothness_penalty
            - lateral_penalty
        )

        return reward

    def _compute_gait_reward(self):
        """Reward for maintaining a good walking gait pattern."""
        # Check which feet are in contact
        feet_in_contact = 0

        for foot_geom_id in self.foot_geom_ids:
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    feet_in_contact += 1
                    break

        # Reward for having 4-6 feet in contact (stable but moving)
        if 4 <= feet_in_contact <= 6:
            return 1.0
        else:
            return 0.5

    def _is_terminated(self):
        """Check if episode should terminate (robot fell or flipped)."""
        # Check height - robot normally stands at 0.134
        height = self.data.qpos[2]
        if height < 0.04:  # More lenient height threshold
            if self.debug:
                print(f"Terminated: Low height {height:.3f}")
            return True

        # Check orientation (if robot flipped)
        # xquat is in [x, y, z, w] order, so index 3 is w
        orientation = self.data.xquat[self.torso_id]
        # However, it seems the robot's "upright" has w≈0, not w≈1
        # So let's check if the robot has rotated significantly from initial
        # The initial orientation seems to be (0.707, 0.707, 0, 0)
        # This suggests a 90-degree rotation is the "normal" state

        # Check if orientation has changed significantly from initial
        # We'll use the magnitude of the first two components
        orientation_norm = np.sqrt(orientation[0] ** 2 + orientation[1] ** 2)
        if orientation_norm < 0.2:  # More lenient orientation check (was 0.3)
            if self.debug:
                print(
                    f"Terminated: Bad orientation norm={orientation_norm:.3f}, quat=({orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}, {orientation[3]:.3f})"
                )
            return True

        # Check if robot has gone too far off course (sideways)
        y_position = abs(self.data.qpos[1])
        if y_position > 10.0:  # More lenient lateral drift (was 5.0)
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


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording videos of the agent at regular intervals.
    """

    def __init__(
        self,
        eval_env,
        render_freq=10000,
        n_eval_episodes=1,
        video_folder="./out/videos/",
        video_length=1000,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = video_folder
        self.video_length = video_length
        self.episode_count = 0

        # Create video folder robustly
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # Record video
            video_name = f"spider_robot_step_{self.n_calls}"
            self._record_video(video_name)
        return True

    def _record_video(self, video_name):
        """Record a video of the agent."""
        try:
            print(f"Recording video: {video_name}")

            # Wrap environment for video recording
            video_path = os.path.join(self.video_folder, video_name)
            vec_env = DummyVecEnv([lambda: self.eval_env])
            vec_env = VecVideoRecorder(
                vec_env,
                video_path,
                record_video_trigger=lambda x: x == 0,
                video_length=self.video_length,
                name_prefix=video_name,
            )

            obs = vec_env.reset()
            episode_rewards = []
            episode_lengths = []
            current_reward = 0
            current_length = 0

            for step in range(self.video_length):
                # Use the trained policy
                action, _ = self.model.predict(
                    obs[0] if isinstance(obs, tuple) else obs, deterministic=True
                )
                obs, reward, done, info = vec_env.step(action)
                current_reward += reward[0]
                current_length += 1

                if done.any():
                    episode_rewards.append(current_reward)
                    episode_lengths.append(current_length)
                    current_reward = 0
                    current_length = 0
                    obs = vec_env.reset()

                    # Log reset info
                    if len(episode_rewards) <= 3:  # Log first few resets
                        print(f"  Episode {len(episode_rewards)} ended at step {step}")
                        if info and len(info) > 0:
                            print(
                                f"    Final height: {info[0].get('height', 'N/A'):.3f}"
                            )
                            print(
                                f"    Final x_position: {info[0].get('x_position', 'N/A'):.3f}"
                            )

            # Log video recording summary
            if episode_rewards:
                print(f"  Video summary: {len(episode_rewards)} episodes")
                print(f"  Average reward: {np.mean(episode_rewards):.2f}")
                print(f"  Average length: {np.mean(episode_lengths):.1f} steps")

            # Clean up
            vec_env.close()

        except Exception as e:
            print(f"Error recording video {video_name}: {e}")
            # Continue training even if video recording fails


def make_env(xml_file, camera_name="bodycam"):
    """Create a wrapped environment for parallel training."""

    def _init():
        env = SpiderRobotEnv(xml_file, render_mode="rgb_array", camera_name=camera_name)
        env = Monitor(env)
        return env

    return _init


def train_spider_robot(
    xml_file,
    total_timesteps=1_000_000,
    record_video=True,  # Re-enable video recording
    camera_name="bodycam",
):
    """Train the spider robot using PPO algorithm."""
    env = DummyVecEnv([make_env(xml_file, camera_name=camera_name)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Video recording
    if record_video:
        env = VecVideoRecorder(
            env,
            video_folder="out/videos/",
            record_video_trigger=lambda x: x % 10000 == 0,
            video_length=1000,
            name_prefix="spider-walk",
        )

    # Create PPO model with custom network architecture
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./out/tensorboard/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save final model
    model.save("./out/spider_robot_final")
    env.close()

    return model


def test_trained_model(xml_file, model_path, camera_name="bodycam"):
    """Test a trained model with rendering."""
    env = SpiderRobotEnv(xml_file, render_mode="human", camera_name=camera_name)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        env.render()

        if terminated or truncated or steps > 1000:
            print(
                f"Episode finished: Total reward = {total_reward:.2f}, Steps = {steps}"
            )
            print(
                f"Final position: x={info['x_position']:.2f}, height={info['height']:.2f}"
            )
            break

    env.close()


if __name__ == "__main__":
    # Example usage
    xml_file = "../robot/SpiderBot.xml"  # Path to your MuJoCo XML file

    # Train the robot with video recording
    print("Starting training with video recording...")
    model = train_spider_robot(
        xml_file, total_timesteps=1_000_000, record_video=True, camera_name="bodycam"
    )

    # Test the trained model
    print("\nTesting trained model...")
    test_trained_model(xml_file, "./out/spider_robot_final", camera_name="bodycam")
