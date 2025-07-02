from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
import numpy as np
from training.joint_config import JOINT_LIMITS


class SpiderEnv(MujocoEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, xml_path, render_mode=None):
        self.joint_names = [name for name, _ in JOINT_LIMITS]
        self.joint_limits = np.array([rng for _, rng in JOINT_LIMITS])
        self.foot_touch_sensor_names = [f"Leg{i}_Tibia_foot" for i in range(1, 9)]
        self.bad_touch_sensor_names = [
            f"Leg{i}_Tibia_bad_touch1" for i in range(1, 9)
        ] + [f"Leg{i}_Tibia_bad_touch2" for i in range(1, 9)]
        self.render_mode = render_mode
        obs_dim = 6 + len(self.joint_names) * 2  # 3 gyro, 3 accel, qpos, qvel
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        super().__init__(
            xml_path,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name="bodycam",
        )

    def _get_obs(self):
        sensors = getattr(self, "_last_sensors", None)
        gyro = sensors["gyro"] if sensors else [0, 0, 0]
        accelerometer = sensors["accelerometer"] if sensors else [0, 0, 0]
        qpos = self.data.qpos[7:]  # exclude base pos/orient
        qvel = self.data.qvel[6:]
        return np.concatenate([gyro, accelerometer, qpos, qvel])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        self._last_sensors = self._read_sensors()
        return self._get_obs()

    def step(self, action):
        # Apply the action using the base class helper
        self.do_simulation(action, self.frame_skip)
        sensors = self._read_sensors()
        self._last_sensors = sensors
        obs = self._get_obs()
        reward = self._compute_reward(sensors)
        terminated = self._is_terminated(sensors)
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _actuator_limit_proximity_penalty(self, qpos):
        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]
        joint_ranges = upper_limits - lower_limits
        threshold = np.maximum(0.05, 0.05 * joint_ranges)  # 5% of range or 0.05 radians
        close_to_lower = (qpos - lower_limits) < threshold
        close_to_upper = (upper_limits - qpos) < threshold
        num_close = np.sum(close_to_lower | close_to_upper)
        penalty = -0.5 * num_close
        return penalty, num_close

    def _compute_reward(self, sensors):
        forward_velocity = self.data.qvel[0]
        upright_bonus = 1.0 if self.data.qpos[2] > 0.1 else 0.0
        energy_penalty = -np.sum(np.square(self.data.ctrl)) * 0.0001
        if sensors["foot_contact"] >= 3:
            foot_contact_reward = 1
        else:
            foot_contact_reward = -2.0
        qpos = self.data.qpos[7:]
        actuator_limit_penalty, num_close = self._actuator_limit_proximity_penalty(qpos)
        within_all_limits_bonus = 0.5 if num_close == 0 else 0.0
        return (
            foot_contact_reward
            + forward_velocity
            + upright_bonus
            + energy_penalty
            + actuator_limit_penalty
            + within_all_limits_bonus
        )

    def _is_terminated(self, sensors):
        z = self.data.qpos[2]
        x, y = self.data.qpos[0], self.data.qpos[1]
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True
        if z < 0.05:
            return True
        if abs(x) > 10.0 or abs(y) > 10.0:
            return True
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
            # 2: GYRO, 3: ACCELEROMETER, 6: TOUCH (from mujoco enums)
            if sensor_type == 2:
                sensors["gyro"] = self.data.sensordata[idx : idx + 3]
            elif sensor_type == 3:
                sensors["accelerometer"] = self.data.sensordata[idx : idx + 3]
            elif sensor_type == 6:
                value = self.data.sensordata[idx]
                positive_value = True if value > 0.0 else False
                # sensor name: self.model.sensor(i).name (if available)
                try:
                    sensor_name = self.model.sensor(i).name
                except Exception:
                    sensor_name = None
                if positive_value and sensor_name:
                    if sensor_name in self.foot_touch_sensor_names:
                        sensors["foot_contact"] += 1
                    elif sensor_name in self.bad_touch_sensor_names:
                        sensors["has_bad_touch"] = True
        return sensors

    def close(self):
        super().close()
