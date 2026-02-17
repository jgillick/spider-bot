"""
Creates a basic spider bot scene for experimentation.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spider bot test scene")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from spider_bot.spider_bot_cfg import SpiderBotCfg

MAX_STEPS = 400

STABLE_FEMUR= 0.5   
STABLE_TIBIA = 0.5

STABLE_POSE = {
    # Hip joints
    "Leg[1-2]_Hip": -1.0,
    "Leg[3-6]_Hip": 1.0,
    "Leg[7-8]_Hip": -1.0,
    # Femur joints
    ".*_Femur": STABLE_FEMUR,
    # Tibia joints
    ".*_Tibia": STABLE_TIBIA,
}

GROUND_POSE = {
    # Hip joints
    "Leg[1-2]_Hip": -1.0,
    "Leg[3-6]_Hip": 1.0,
    "Leg[7-8]_Hip": -1.0,
    # Femur joints
    ".*_Femur": 0.01,
    # Tibia joints
    ".*_Tibia": 2.6,
}

@configclass
class SpiderSceneCfg(InteractiveSceneCfg):
    """Configuration for a test spider bot scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot = SpiderBotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SpiderBotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.12),
            joint_pos=GROUND_POSE,
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/.*", 
        history_length=3, 
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    joint_names = robot.data.joint_names
    
    # Get initial joint positions
    initial_joint_pos = robot.data.default_joint_pos.clone()
    target_joint_pos = initial_joint_pos.clone()


    # Simulation loop
    count = 0
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and count < MAX_STEPS:
        # Reset the state
        # if count == 0:
        #     root_state = robot.data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     robot.write_root_pose_to_sim(root_state[:, :7])
        #     robot.write_root_velocity_to_sim(root_state[:, 7:])
        #     joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        #     robot.write_joint_state_to_sim(joint_pos, joint_vel)
        #     scene.reset()

        # Standup 
        if count == 100:
            print("Standing up (1)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg1_Tibia", "Leg2_Tibia", "Leg3_Tibia", "Leg4_Tibia", "Leg5_Tibia", "Leg6_Tibia", "Leg7_Tibia", "Leg8_Tibia"):
                    target_joint_pos[0][i] = STABLE_TIBIA
        if count == 110:
            print("Standing up (2)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg1_Femur", "Leg2_Femur", "Leg3_Femur", "Leg4_Femur", "Leg5_Femur", "Leg6_Femur", "Leg7_Femur", "Leg8_Femur"):
                    target_joint_pos[0][i] = STABLE_FEMUR

        if count % 10 == 0:
            print(f"[{count}] Height: {robot.data.root_pos_w[:, 2]}")

        robot.set_joint_position_target(target_joint_pos)        
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # self.sim.dt = 0.002
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    
    # Design scene
    scene_cfg = SpiderSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    print("-"*100)
    print("Starting interactive simulation...")
    print(f"Max steps: {MAX_STEPS}")
    print("-"*100)
    run_simulator(sim, scene)
    print("-"*100)
    print("Interactive simulation complete")
    print("-"*100)


if __name__ == "__main__":
    # run the main function
    main()
    sys.exit(0)