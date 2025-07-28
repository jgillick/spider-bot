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

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.assets import RigidObject

from spider_bot_cfg import SpiderBotCfg

STABLE_FEMUR= 0.9   
STABLE_TIBIA = -0.9

STABLE_POSE = {
    # Hip joints
    ".*Leg1_Hip": 1.0,
    ".*Leg2_Hip": 1.0,
    ".*Leg3_Hip": -1.0,
    ".*Leg4_Hip": -1.0,
    ".*Leg5_Hip": -1.0,
    ".*Leg6_Hip": -1.0,
    ".*Leg7_Hip": 1.0,
    ".*Leg8_Hip": 1.0,
    # Femur joints
    ".*_Femur": STABLE_FEMUR,
    # Tibia joints
    ".*_Tibia": STABLE_TIBIA,
}

FLAT_OUT_POSE = {
    # Hip joints
    ".*Leg1_Hip": 1.0,
    ".*Leg2_Hip": 1.0,
    ".*Leg3_Hip": -1.0,
    ".*Leg4_Hip": -1.0,
    ".*Leg5_Hip": -1.0,
    ".*Leg6_Hip": -1.0,
    ".*Leg7_Hip": 1.0,
    ".*Leg8_Hip": 1.0,
    # Femur joints
    ".*_Femur": 0.01,
    # Tibia joints
    ".*_Tibia": -2.6,
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
            pos=(0.0, 0.0, 0.1),
            joint_pos=FLAT_OUT_POSE,
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/.*", 
        history_length=3, 
    )

def log_tibia_forces(robot: RigidObject):
    """Log the forces on the tibia joints."""
    body_wrenches = robot.data.body_incoming_joint_wrench_b
    torque_list = []
    for i in range(1, 9):
        body_name = f"Leg{i}_Tibia_Leg"
        idx = robot.find_bodies(body_name)[0][0]
        wrench = body_wrenches[:, idx, :]
        torques = wrench[:, :3:6]
        load_torque = torch.round(torques[:,0], decimals=1).item()
        torque_list.append(f"{load_torque:.2f}")
    
    print(f"Tibia torques: {torque_list}")

def log_bad_touch_forces(contact_sensor):
    bad_touch_sensors, _ = contact_sensor.find_bodies(".*_BadTouch")
    
    forces = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w[:, bad_touch_sensors], dim=-1
    )[0]
    pretty = [f"{force.item():.1f}" for force in forces]
    print(f"Bad touch forces: {pretty}")

def log_foot_forces(contact_sensor):
    foot_sensors, _ = contact_sensor.find_bodies(".*_Tibia_Foot")
    
    forces = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w[:, foot_sensors], dim=-1
    )[0]
    pretty = [f"{force.item():.1f}" for force in forces]
    print(f"Foot forces: {pretty}")

def log_forces(contact_sensor):
    net_contact_forces = contact_sensor.data.net_forces_w[0]
    for i, vector in enumerate(net_contact_forces):
        force = torch.linalg.vector_norm(vector, dim=-1).item()
        if abs(force) > 0.001:
            print(f"{contact_sensor.body_names[i]}: {force}")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    joint_names = robot.data.joint_names

    hip1_join_id = joint_names.index("Leg1_Hip")
    hip2_join_id = joint_names.index("Leg2_Hip")
    femur1_joint_id = joint_names.index("Leg2_Femur")
    
    # Get initial joint positions
    initial_joint_pos, initial_joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    target_joint_pos = initial_joint_pos.clone()


    # Simulation loop
    count = 0
    last_height = -1.0
    last_angle = -1.0
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # Reset
        # if count % 200 == 0:
        #     print("[INFO]: Resetting robot state...")
        #     count = 0
        #     root_state = robot.data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     robot.write_root_pose_to_sim(root_state[:, :7])
        #     robot.write_root_velocity_to_sim(root_state[:, 7:])
        #     robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
        #     scene.reset()
        

        # angle = torch.acos(-robot.data.projected_gravity_b[:, 2]).abs().item()
        # if angle != last_angle:
        #     print(f"angle: {angle}")
        # last_angle = angle

        # Standup 
        if count == 100:
            print("Standing up (1)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg1_Tibia", "Leg4_Tibia", "Leg5_Tibia", "Leg8_Tibia"):
                    target_joint_pos[0][i] = STABLE_TIBIA
        if count == 110:
            print("Standing up (2)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg1_Femur", "Leg4_Femur", "Leg5_Femur", "Leg8_Femur"):
                    target_joint_pos[0][i] = STABLE_FEMUR
                if name in ("Leg2_Tibia", "Leg3_Tibia", "Leg6_Tibia", "Leg7_Tibia"):
                    target_joint_pos[0][i] = 0.01
        if count == 200:
            print("Collapse (1)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg2_Femur", "Leg3_Femur", "Leg6_Femur", "Leg7_Femur"):
                    target_joint_pos[0][i] = 0.99
        if count == 250:
            print("Collapse (2)...")
            for i, name in enumerate(joint_names):
                if name in ("Leg1_Femur", "Leg4_Femur", "Leg5_Femur", "Leg8_Femur"):
                    target_joint_pos[0][i] = 0.01
        
        if count % 10 == 0:
            log_bad_touch_forces(contact_sensor)
            log_foot_forces(contact_sensor)  
            # log_forces(contact_sensor)
        
        robot.set_joint_position_target(target_joint_pos)
        
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
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
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()