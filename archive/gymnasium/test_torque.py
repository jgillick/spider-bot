"""
Test script to demonstrate torque behavior when robot is reset to INITIAL_JOINT_POSITIONS.
Shows what happens with different action values and whether torque is automatically applied.
"""

import numpy as np
import os

from .environment import SpiderRobotEnv

INITIAL_JOINT_POSITIONS = (
    -1.0,  # Leg 1 - Hip
    0.0,  # Leg 1 - Femur
    2.7,  # Leg 1 - Tibia
    -1.0,  # Leg 2 - Hip
    0.0,  # Leg 2 - Femur
    2.7,  # Leg 2 - Tibia
    0.7,  # Leg 3 - Hip
    0.0,  # Leg 3 - Femur
    2.7,  # Leg 3 - Tibia
    1.0,  # Leg 4 - Hip
    0.0,  # Leg 4 - Femur
    2.7,  # Leg 4 - Tibia
    1.0,  # Leg 5 - Hip
    0.0,  # Leg 5 - Femur
    2.7,  # Leg 5 - Tibia
    1.0,  # Leg 6 - Hip
    0.0,  # Leg 6 - Femur
    2.7,  # Leg 6 - Tibia
    -1.0,  # Leg 7 - Hip
    0.0,  # Leg 7 - Femur
    2.7,  # Leg 7 - Tibia
    -1.0,  # Leg 8 - Hip
    0.0,  # Leg 8 - Femur
    2.7,  # Leg 8 - Tibia
)


def test_torque_behavior():
    """Test different action scenarios to understand torque behavior."""

    # Initialize environment
    xml_file = os.path.join(os.path.dirname(__file__), "../robot/SpiderBot.xml")
    env = SpiderRobotEnv(xml_file=xml_file, render_mode="human", width=1200, height=800)

    print("\n📊 Test 2: Actions to Maintain Initial Positions")
    print("-" * 50)

    obs = env.reset()
    initial_height = env.data.qpos[2]

    print(f"Initial height: {initial_height:.3f}m")
    print("\nSimulating with MAINTAIN actions...")

    # Calculate normalized actions needed to maintain initial positions
    maintain_actions = []
    qpos = env.init_qpos.copy()
    first_joint_idx = len(qpos) - 24
    for i, (low, high) in enumerate(env.joint_ranges):
        current_pos = INITIAL_JOINT_POSITIONS[i]

        # Set starting position
        qpos[first_joint_idx + i] = current_pos

        # Normalize to -1 to 1
        normalized_action = (2 * (current_pos - low) / (high - low)) - 1
        maintain_actions.append(normalized_action)
    maintain_actions = np.array(maintain_actions, dtype=np.float32)

    # Set initial model position/velocity
    qpos[2] = 0.08
    env.set_state(qpos, env.init_qvel)

    # print(
    #     f"{'Pos':<15} | {'Contact':<15} | {'Action Torque':<15} | {'Force':<15} | {'Control':<15} | {'QFRC':<15}"
    # )
    # joint_idx = 1
    # joint_id = 1 + joint_idx

    print(
        f"{'Stage':<5} | {'Feet':<5} | {'Torque In':<15} | {'Femur Out':<10} | {'Tibia Out':<10} | {'Combined':<10}"
    )
    print("-" * 60)
    stage = 0
    joint_start = 6
    for step in range(300):
        if step == 100:
            stage += 1
            for i, pos in enumerate(maintain_actions):
                if (i - 1) % 3 == 0:
                    maintain_actions[i] = 1.5
                if (i + 1) % 3 == 0:
                    maintain_actions[i] = -0.5

        obs, reward, done, truncated, info = env.step(maintain_actions)
        foot_contacts = np.sum(env.last_foot_contacts)
        env.render()

        femur_join_id = joint_start + 1
        tibia_join_id = joint_start + 2
        femur_in = env.last_torques[1]
        tibia_in = env.last_torques[2]
        femur_out = (
            # env.data.qfrc_actuator[femur_join_id]
            +env.data.qfrc_bias[femur_join_id]
            + env.data.qfrc_constraint[femur_join_id]
        )
        tibia_out = (
            # env.data.qfrc_actuator[tibia_join_id]
            +env.data.qfrc_bias[tibia_join_id]
            + env.data.qfrc_constraint[tibia_join_id]
        )
        combined_in = f"{femur_in:.2f} / {tibia_in:.2f}"
        combined_out = abs(femur_out) + abs(tibia_out)

        print(
            f"{stage:<5} | {foot_contacts:<5.0f} | {combined_in:<15} | {femur_out:<10.2f} | {tibia_out:<10.2f} | {combined_out:<10.2f}"
        )

        # Use sensors: jointactuatorfrc, actuatorfrc
        # torque_sensor = env.data.sensordata[-2:]
        # print(
        #     f"{maintain_actions[joint_idx+1]:<15.2f} | {foot_contacts:<15.2f} | {env.last_torques[joint_idx]:<15.2f} | {env.data.actuator_force[joint_id]:<15.2f} | {env.data.ctrl[joint_id]:<15.2f} | {env.data.qfrc_actuator[joint_id]:<15.2f}"
        # )

        if done:
            print(f"❌ Episode terminated at step {step}")
            break

    env.close()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_torque_behavior()
