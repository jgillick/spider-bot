import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer
from training.joint_config import JOINT_LIMITS


class SpiderTorqueEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.render_mode = render_mode

        self.joint_names = [name for name, _ in JOINT_LIMITS]
        self.joint_limits = np.array([rng for _, rng in JOINT_LIMITS])
        self.num_actuators = self.model.nu

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32
        )

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Scale normalized actuator action to [-10, 10]
        scaled_torque = 10.0 * np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = scaled_torque

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # IMU data: assuming sensors are present (gyro + accelerometer)
        imu = self.data.sensordata[:6] if self.model.nsensordata >= 6 else np.zeros(6)
        qpos = self.data.qpos[7:]  # exclude base pos/orient
        qvel = self.data.qvel[6:]
        return np.concatenate([imu, qpos, qvel])

    def _compute_reward(self):
        forward_velocity = self.data.qvel[0]
        upright_bonus = 1.0 if self.data.qpos[2] > 0.05 else 0.0
        energy_penalty = -np.sum(np.square(self.data.ctrl)) * 0.001
        return forward_velocity + upright_bonus + energy_penalty

    def _is_terminated(self):
        z = self.data.qpos[2]
        x, y = self.data.qpos[0], self.data.qpos[1]

        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True
        if z < 0.05:
            return True
        if abs(x) > 10.0 or abs(y) > 10.0:
            return True
        return False

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            width, height = 800, 600
            if not hasattr(self, "rgb_renderer"):
                self.rgb_renderer = mujoco.Renderer(
                    self.model, height=height, width=width
                )
            self.rgb_renderer.update_scene(self.data, camera="bodycam")
            return self.rgb_renderer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
