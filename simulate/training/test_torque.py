"""
Test script to demonstrate torque behavior when robot is reset to INITIAL_JOINT_POSITIONS.
Shows what happens with different action values and whether torque is automatically applied.
"""

import numpy as np
import os

from .environment import SpiderRobotEnv, INITIAL_JOINT_POSITIONS


def test_torque_behavior():
    """Test different action scenarios to understand torque behavior."""

    # Initialize environment
    xml_file = os.path.join(os.path.dirname(__file__), "../robot/SpiderBot.xml")
    env = SpiderRobotEnv(xml_file=xml_file, render_mode="human")

    print("\n📊 Test 2: Actions to Maintain Initial Positions")
    print("-" * 50)

    obs = env.reset()
    initial_height = env.data.qpos[2]

    print(f"Initial height: {initial_height:.3f}m")
    print("\nSimulating with MAINTAIN actions...")

    # Calculate normalized actions needed to maintain initial positions
    maintain_actions = []
    for i, (low, high) in enumerate(env.joint_ranges):
        # Reverse the scaling formula: action = (2 * (pos - low) / (high - low)) - 1
        current_pos = INITIAL_JOINT_POSITIONS[i]
        normalized_action = (2 * (current_pos - low) / (high - low)) - 1
        maintain_actions.append(normalized_action)
    maintain_actions = np.array(maintain_actions, dtype=np.float32)

    for step in range(100):
        obs, reward, done, truncated, info = env.step(maintain_actions)
        env.render()

        if done:
            print(f"❌ Episode terminated at step {step}")
            break

    env.close()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_torque_behavior()
