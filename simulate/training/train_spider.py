import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from training.spider_torque_env import SpiderTorqueEnv

# Path to your MJCF XML
XML_PATH = "../robot/SpiderBot.xml"


def make_env():
    def _init():
        env = SpiderTorqueEnv(XML_PATH, render_mode="rgb_array")
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Optional: record videos
    env = VecVideoRecorder(
        env,
        video_folder="logs/videos/",
        record_video_trigger=lambda x: x % 10000 == 0,
        video_length=1000,
        name_prefix="spider-walk",
    )

    # Define model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Train
    model.learn(total_timesteps=1_000_000)

    # Save model
    model.save(os.path.join(log_dir, "ppo_spider"))
    env.close()
