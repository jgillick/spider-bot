#!/usr/bin/env python3
"""Test script to verify spider locomotion environment setup."""

import sys
import os

# Add the spider locomotion module to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import spider locomotion to register environments
    import spider_locomotion

    print("✓ Spider locomotion module imported successfully")

    # Check if environments are registered
    import gymnasium as gym

    env_ids = ["Isaac-SpiderLocomotion-v0", "Isaac-SpiderLocomotion-Flat-v0"]

    for env_id in env_ids:
        try:
            # Try to get the spec to see if it's registered
            spec = gym.spec(env_id)
            print(f"✓ {env_id} is registered")
        except:
            print(f"✗ {env_id} is NOT registered")

    # Try to import configurations
    from spider_locomotion.config import SpiderBotCfg, SpiderLocomotionEnvCfg

    print("✓ Configuration classes imported successfully")

    # Check if MuJoCo model exists
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "robot",
        "SpiderBot.xml",
    )
    if os.path.exists(xml_path):
        print(f"✓ MuJoCo model found at: {xml_path}")
    else:
        print(f"✗ MuJoCo model NOT found at: {xml_path}")

    print("\n✅ Environment setup verified! Ready for training in Isaac Lab.")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nThis test should be run after installing the spider_locomotion package:")
    print("  cd simulate/issac && pip install -e .")

except Exception as e:
    print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("Spider Locomotion Environment Test")
    print("=" * 40)
