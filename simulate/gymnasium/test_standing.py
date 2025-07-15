"""Test rewards for standing still vs simple walking motions"""

import numpy as np
from .environment import SpiderRobotEnv, INITIAL_JOINT_POSITIONS
import os

# Get the robot XML path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FILE = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


FALLING_POSITIONS = (
    -1.0,  # Leg 1 - Hip
    0.0,  # Leg 1 - Femur
    1.0,  # Leg 1 - Tibia
    -1.0,  # Leg 2 - Hip
    0.0,  # Leg 2 - Femur
    1.0,  # Leg 2 - Tibia
    1.0,  # Leg 3 - Hip
    0.0,  # Leg 3 - Femur
    1.0,  # Leg 3 - Tibia
    1.0,  # Leg 4 - Hip
    0.0,  # Leg 4 - Femur
    1.0,  # Leg 4 - Tibia
    1.0,  # Leg 5 - Hip
    0.0,  # Leg 5 - Femur
    1.0,  # Leg 5 - Tibia
    1.0,  # Leg 6 - Hip
    0.0,  # Leg 6 - Femur
    1.0,  # Leg 6 - Tibia
    -1.0,  # Leg 7 - Hip
    0.0,  # Leg 7 - Femur
    1.0,  # Leg 7 - Tibia
    -1.0,  # Leg 8 - Hip
    0.0,  # Leg 8 - Femur
    1.0,  # Leg 8 - Tibia
)


def test_reward_breakdown():
    print("🕷️ Testing Reward Components")
    print("=" * 50)

    env = SpiderRobotEnv(XML_FILE, render_mode="human", width=1200, height=800)

    # Test 1: Zero actions (standing still)
    print("\n📊 Test 1: Standing still (zero actions)")
    obs = env.reset_model()

    maintain_actions = []
    for i, (low, high) in enumerate(env.joint_ranges):
        current_pos = INITIAL_JOINT_POSITIONS[i]
        normalized_action = (2 * (current_pos - low) / (high - low)) - 1
        maintain_actions.append(normalized_action)
    maintain_actions = np.array(maintain_actions, dtype=np.float32)

    total_reward = 0
    for step in range(100):
        obs, reward, done, truncated, info = env.step(maintain_actions)
        env.render()
        total_reward += reward

        if step % 10 == 0:
            print(f"  Step {step}: reward={reward:.3f}, height={env.data.qpos[2]:.3f}")
            # Get detailed reward breakdown
            foot_contacts = env._get_foot_contacts()
            print(f"    Feet on ground: {np.sum(foot_contacts)}")

    print(f"  Average reward over 10 steps: {total_reward/10:.3f}")

    env.close()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_reward_breakdown()
