#!/usr/bin/env python3
"""
Test script for the Spider Robot Environment
"""

import numpy as np
from env import SpiderRobotEnv


def test_environment():
    """Test the spider robot environment."""
    xml_file = "../robot/SpiderBot.xml"

    print("Creating environment...")
    env = SpiderRobotEnv(xml_file, render_mode=None, debug=True)

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    print("Running a few steps...")
    total_reward = 0
    for step in range(10):
        # Random actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step}: reward={reward:.3f}, height={info['height']:.3f}, terminated={terminated}"
        )

        if terminated:
            print("Episode terminated early!")
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_environment()
