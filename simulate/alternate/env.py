"""
Spider Robot Reinforcement Learning Environment and Training
Uses Gymnasium and MuJoCo for physics simulation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import seeding
import mujoco
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
import os


class SpiderRobotEnv(MujocoEnv):
    """
    Custom Gymnasium environment for training a spider robot to walk.

    The robot has:
    - 8 legs
    - 3 actuators per leg (24 total actuators)
    - IMU sensor for orientation/acceleration feedback
    - Position-controlled torque motors with limits
    """

    def __init__(self, xml_file, frame_skip=5, render_mode=None, **kwargs):
        # Initialize tracking variables
        self.previous_x_position = 0
        self.steps_taken = 0
        self.fall_count = 0

        # Control parameters
        self.max_torque = 10.0  # 10N max torque
        self.position_gain = 10.0  # P gain for position control
        self.velocity_gain = 0.5  # D gain for position control

        # Initialize MuJoCo environment
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_obs_dim(xml_file),),
            dtype=np.float64,
        )

        # Set metadata with correct render_fps
        # First load the model to get the timestep
        temp_model = mujoco.MjModel.from_xml_path(xml_file)
        dt = temp_model.opt.timestep * frame_skip
        render_fps = int(np.round(1.0 / dt))

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

    def _get_obs_dim(self, xml_file):
        """Calculate observation dimension based on the model."""
        # Temporarily load model to get dimensions
        model = mujoco.MjModel.from_xml_path(xml_file)
        # Observation includes:
        # - Joint positions (nq)
        # - Joint velocities (nv)
        # - IMU data (orientation quaternion + angular velocity + linear acceleration = 4+3+3)
        # - Body position and orientation of torso (7)
        # - Contact forces for each foot (8)
        return model.nq + model.nv + 10 + 7 + 8

    def step(self, action):
        """Execute one timestep of the environment dynamics."""
        # Clip actions to actuator limits
        action = np.clip(action, self.action_space.low, self.action_space.high)

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

        # Add small random perturbations for robustness
        qpos[7:] += self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq - 7)
        qvel[6:] += self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv - 6)

        self.set_state(qpos, qvel)

        # Reset tracking variables
        self.previous_x_position = self.data.qpos[0]
        self.steps_taken = 0

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
                    # Get contact force magnitude
                    c_array = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, j, c_array)
                    contact_forces[idx] = np.linalg.norm(
                        c_array[:3]
                    )  # Only normal force

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
        # Forward progress reward
        current_x = self.data.qpos[0]
        forward_reward = (current_x - self.previous_x_position) / self.dt
        self.previous_x_position = current_x

        # Height reward (encourage maintaining height)
        target_height = 0.3  # Adjust based on your robot
        height = self.data.qpos[2]
        height_reward = np.exp(-5 * (height - target_height) ** 2)

        # Orientation reward (stay upright)
        orientation = self.data.xquat[self.torso_id]
        upright_reward = orientation[3] ** 2  # w component of quaternion

        # Energy efficiency penalty
        energy_penalty = 0.001 * np.sum(np.square(self.data.ctrl))

        # Foot contact reward (encourage alternating gait)
        contact_pattern_reward = self._compute_gait_reward()

        # Smoothness reward (penalize jerky movements)
        smoothness_penalty = 0.01 * np.sum(np.square(self.data.qvel[6:]))

        # Total reward
        reward = (
            5.0 * forward_reward
            + 2.0 * height_reward
            + 1.0 * upright_reward
            + 1.0 * contact_pattern_reward
            - energy_penalty
            - smoothness_penalty
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
        # Check height
        height = self.data.qpos[2]
        if height < 0.1:  # Adjust threshold based on your robot
            return True

        # Check orientation (if robot flipped)
        orientation = self.data.xquat[self.torso_id]
        if orientation[3] < 0.5:  # w component < 0.5 means tilted > 60 degrees
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
        video_folder="./spider_robot_videos/",
        video_length=1000,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = video_folder
        self.video_length = video_length
        self.episode_count = 0

        # Create video folder
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # Record video
            video_name = f"spider_robot_step_{self.n_calls}"
            self._record_video(video_name)
        return True

    def _record_video(self, video_name):
        """Record a video of the agent."""
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
        for _ in range(self.video_length):
            # Use the trained policy
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            if done.any():
                obs = vec_env.reset()

        # Clean up
        vec_env.close()


def make_env(xml_file, rank, seed=0):
    """Create a wrapped environment for parallel training."""

    def _init():
        env = SpiderRobotEnv(xml_file, render_mode=None)
        env.reset(seed=seed + rank)
        return env

    return _init


def train_spider_robot(
    xml_file, total_timesteps=1_000_000, n_envs=4, record_video=True
):
    """Train the spider robot using PPO algorithm."""
    # Create parallel environments
    env = SubprocVecEnv([make_env(xml_file, i) for i in range(n_envs)])

    # Create evaluation environment
    eval_env = SpiderRobotEnv(xml_file, render_mode=None)

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./spider_robot_checkpoints/",
        name_prefix="spider_robot_model",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./spider_robot_best_model/",
        log_path="./spider_robot_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Video recording callback
    if record_video:
        # Create a separate environment for video recording
        video_env = SpiderRobotEnv(xml_file, render_mode="rgb_array")
        video_callback = VideoRecorderCallback(
            video_env,
            render_freq=25000,  # Record video every 25k steps
            n_eval_episodes=1,
            video_folder="./spider_robot_videos/",
            video_length=1000,  # 1000 steps per video
        )
        callbacks.append(video_callback)

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
        tensorboard_log="./spider_robot_tensorboard/",
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save final model
    model.save("spider_robot_final")

    # Close video environment if used
    if record_video:
        video_env.close()

    return model


def test_trained_model(xml_file, model_path):
    """Test a trained model with rendering."""
    env = SpiderRobotEnv(xml_file, render_mode="human")
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
    model = train_spider_robot(xml_file, total_timesteps=1_000_000, record_video=True)

    # Test the trained model
    print("\nTesting trained model...")
    test_trained_model(xml_file, "spider_robot_final")
