"""
Training script for Spider Robot using PPO
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecVideoRecorder,
)
from .environment import make_env


DEBUG = False
VIDEO = True

TRAINING_STEPS = 2_000_000
VIDEO_EVERY_N_STEPS = 50_000
VIDEO_LENGTH = 1_000  # in steps

OUT_DIR = "../out"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def train_spider_robot(xml_file):
    """Train the spider robot using PPO algorithm."""

    # Get absolute output path, relative to this script
    out_path = os.path.abspath(os.path.join(THIS_DIR, OUT_DIR))

    # Create output directories
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(f"{out_path}/videos", exist_ok=True)
    os.makedirs(f"{out_path}/tensorboard", exist_ok=True)

    # Create environment
    env = DummyVecEnv([make_env(xml_file, debug=DEBUG)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Video recording
    if VIDEO:
        env = VecVideoRecorder(
            env,
            video_folder=f"{out_path}/videos/",
            record_video_trigger=lambda x: x % VIDEO_EVERY_N_STEPS == 0,
            video_length=VIDEO_LENGTH,
        )

    # Create PPO model with custom network architecture
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=f"{out_path}/tensorboard/",
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    )

    # Train the model
    print(f"Starting training for {TRAINING_STEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_STEPS)

    # Save final model and normalization stats
    model.save(f"{out_path}/spider_robot_final")
    env.save(f"{out_path}/vec_normalize.pkl")
    env.close()

    print(f"Training complete! Model saved to {out_path}/spider_robot_final")
    return model


if __name__ == "__main__":
    # Example usage
    xml_file = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))

    # Train the robot with video recording
    print("=== Spider Robot Training ===")
    model = train_spider_robot(xml_file)
