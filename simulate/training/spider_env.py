import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from training.joint_config import JOINT_LIMITS


class SpiderEnv(gym.Env):
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

        # Touch sensor names
        self.foot_touch_sensor_names = [f"Leg{i}_Tibia_foot" for i in range(1, 9)]
        self.bad_touch_sensor_names = [
            f"Leg{i}_Tibia_bad_touch1" for i in range(1, 9)
        ] + [f"Leg{i}_Tibia_bad_touch2" for i in range(1, 9)]

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

        sensors = self._read_sensors()
        obs = self._get_obs(sensors)
        reward = self._compute_reward(sensors)
        terminated = self._is_terminated(sensors)
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self, sensors=None):
        gyro = sensors["gyro"] if sensors else [0, 0, 0]
        accelerometer = sensors["accelerometer"] if sensors else [0, 0, 0]
        qpos = self.data.qpos[7:]  # exclude base pos/orient
        qvel = self.data.qvel[6:]
        return np.concatenate([gyro, accelerometer, qpos, qvel])

    def _compute_reward(self, sensors):
        # forward velocity
        forward_velocity = self.data.qvel[0]

        # The robot is off the ground
        upright_bonus = 1.0 if self.data.qpos[2] > 0.1 else 0.0

        # How much energy is being used
        energy_penalty = -np.sum(np.square(self.data.ctrl)) * 0.0001

        # Reward based on number of feet in contact with the ground
        if sensors["foot_contact"] >= 3:
            foot_contact_reward = 1
        else:
            foot_contact_reward = -2.0  # penalize unstable stance
        return foot_contact_reward + forward_velocity + upright_bonus + energy_penalty

    def _is_terminated(self, sensors):
        # Terminate if it fell over or left the arena
        z = self.data.qpos[2]
        x, y = self.data.qpos[0], self.data.qpos[1]
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True
        if z < 0.05:
            return True
        if abs(x) > 10.0 or abs(y) > 10.0:
            return True

        # Terminate if any bad touch sensor is active
        if sensors["has_bad_touch"]:
            return True
        return False

    def _read_sensors(self):
        sensors = {
            "gyro": [0, 0, 0],
            "accelerometer": [0, 0, 0],
            "foot_contact": 0,
            "has_bad_touch": False,
        }
        for i in range(self.model.nsensor):
            idx = self.model.sensor_adr[i]
            sensor_type = self.model.sensor_type[i]
            if sensor_type == mujoco.mjtSensor.mjSENS_GYRO:
                sensors["gyro"] = self.data.sensordata[idx : idx + 3]
            elif sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
                sensors["accelerometer"] = self.data.sensordata[idx : idx + 3]
            elif sensor_type == mujoco.mjtSensor.mjSENS_TOUCH:
                value = self.data.sensordata[idx]
                positive_value = True if value > 0.0 else False
                sensor_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_SENSOR, i
                )
                if positive_value:
                    if sensor_name in self.foot_touch_sensor_names:
                        sensors["foot_contact"] += 1
                    elif sensor_name in self.bad_touch_sensor_names:
                        sensors["has_bad_touch"] = True
        return sensors

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
