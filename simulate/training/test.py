"""
Testing script for trained Spider Robot models
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from .environment import SpiderRobotEnv


def test_trained_model(xml_file, model_path, camera_name="bodycam", num_episodes=5):
    """Test a trained model and record videos."""

    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create test environment
    env = SpiderRobotEnv(xml_file, render_mode="rgb_array", camera_name=camera_name)

    # Note: VecNormalize loading is complex for single environments
    # For now, we'll test without normalization stats
    print("Testing without normalization stats (may affect performance)")

    # Evaluate the model
    print(f"Evaluating model over {num_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, deterministic=True
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run a few episodes with rendering
    print("Running episodes with rendering...")
    for episode in range(min(3, num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()
    return mean_reward, std_reward


def analyze_model_behavior(xml_file, model_path, camera_name="bodycam"):
    """Analyze the behavior of a trained model in detail."""

    # Load the trained model
    model = PPO.load(model_path)

    # Create environment with debug mode
    env = SpiderRobotEnv(
        xml_file, render_mode="human", camera_name=camera_name, debug=True
    )

    # Note: VecNormalize loading is complex for single environments
    # For now, we'll analyze without normalization stats
    print("Analyzing without normalization stats")

    print("Running detailed analysis (press 'q' to quit)...")

    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Print detailed info every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
            print(f"  Position: ({info['x_position']:.3f}, {info['height']:.3f})")
            print(f"  Speed: {info['forward_speed']:.3f}")
            print(f"  Energy: {info['energy_used']:.3f}")

        if terminated or truncated:
            print(
                f"Episode ended after {steps} steps with total reward {total_reward:.3f}"
            )
            break

    env.close()


if __name__ == "__main__":
    # Example usage
    xml_file = "../robot/SpiderBot.xml"
    model_path = "./out/spider_robot_final"

    if os.path.exists(model_path):
        print("=== Testing Trained Model ===")
        mean_reward, std_reward = test_trained_model(xml_file, model_path)

        # Uncomment to run detailed analysis with rendering
        # analyze_model_behavior(xml_file, model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
