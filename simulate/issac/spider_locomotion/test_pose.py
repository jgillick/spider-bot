import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from spider_bot_cfg import SpiderBotCfg


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Origin 
    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)

    # Robot
    model_cfg = SpiderBotCfg(
        prim_path="/World/Origin.*/Robot",
        init_state=SpiderBotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16),
            joint_pos={
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
                ".*_Femur": 0.9,
                # Tibia joints
                ".*_Tibia": -1.0,
            },
        ),
    )
    model = Articulation(cfg=model_cfg)

    # return the scene information
    scene_entities = {"robot": model}
    return scene_entities, [origin]


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

    # Simulation loop
    last_height = -1.0
    while simulation_app.is_running():
        # Reset
        if count % 200 == 0:
            print("[INFO]: Resetting robot state...")
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
        
        # Log height
        height = round(robot.data.root_pos_w[:, 2].item(), 3)
        if count % 10 == 0 and height != last_height:
            print(f"robot height: {height}")
        last_height = height

        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()