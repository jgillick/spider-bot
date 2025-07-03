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
from environment import make_env


DEBUG = False
VIDEO = True

TRAINING_STEPS = 500_000
VIDEO_EVERY_N_STEPS = 25000
VIDEO_LENGTH = 1000  # in steps

OUT_DIR = "../logs"


def train_spider_robot(xml_file):
    """Train the spider robot using PPO algorithm."""

    # Get absolute output path, relative to this script
    this_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.abspath(os.path.join(this_dir, OUT_DIR))

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
        ent_coef=0.01,  # Exploration
        verbose=1,
        tensorboard_log=f"{out_path}/tensorboard/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
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
    xml_file = "../robot/SpiderBot.xml"  # Path to your MuJoCo XML file

    # Train the robot with video recording
    print("=== Spider Robot Training ===")
    model = train_spider_robot(xml_file)
