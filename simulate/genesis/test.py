import os
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.sensors.raycaster.patterns import GridPattern

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))

INITIAL_BODY_POSITION = [0.0, 0.0, 0.14]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
STABLE_FEMUR = 0.5
STABLE_TIBIA = 0.6

STABLE_POS = {
    "Leg1_Hip": -1.0,
    "Leg1_Femur": STABLE_FEMUR,
    "Leg1_Tibia": STABLE_TIBIA,
    "Leg2_Hip": -1.0,
    "Leg2_Femur": STABLE_FEMUR,
    "Leg2_Tibia": STABLE_TIBIA,
    "Leg3_Hip": 1.0,
    "Leg3_Femur": STABLE_FEMUR,
    "Leg3_Tibia": STABLE_TIBIA,
    "Leg4_Hip": 1.0,
    "Leg4_Femur": STABLE_FEMUR,
    "Leg4_Tibia": STABLE_TIBIA,
    "Leg5_Hip": 1.0,
    "Leg5_Femur": STABLE_FEMUR,
    "Leg5_Tibia": STABLE_TIBIA,
    "Leg6_Hip": 1.0,
    "Leg6_Femur": STABLE_FEMUR,
    "Leg6_Tibia": STABLE_TIBIA,
    "Leg7_Hip": -1.0,
    "Leg7_Femur": STABLE_FEMUR,
    "Leg7_Tibia": STABLE_TIBIA,
    "Leg8_Hip": -1.0,
    "Leg8_Femur": STABLE_FEMUR,
    "Leg8_Tibia": STABLE_TIBIA,
}

GROUND_FEMUR = 0.01
GROUND_TIBIA = 2.6
GROUND_POSE = {
    "Leg1_Hip": -1.0,
    "Leg1_Femur": GROUND_FEMUR,
    "Leg1_Tibia": GROUND_TIBIA,
    "Leg2_Hip": -1.0,
    "Leg2_Femur": GROUND_FEMUR,
    "Leg2_Tibia": GROUND_TIBIA,
    "Leg3_Hip": 1.0,
    "Leg3_Femur": GROUND_FEMUR,
    "Leg3_Tibia": GROUND_TIBIA,
    "Leg4_Hip": 1.0,
    "Leg4_Femur": GROUND_FEMUR,
    "Leg4_Tibia": GROUND_TIBIA,
    "Leg5_Hip": 1.0,
    "Leg5_Femur": GROUND_FEMUR,
    "Leg5_Tibia": GROUND_TIBIA,
    "Leg6_Hip": 1.0,
    "Leg6_Femur": GROUND_FEMUR,
    "Leg6_Tibia": GROUND_TIBIA,
    "Leg7_Hip": -1.0,
    "Leg7_Femur": GROUND_FEMUR,
    "Leg7_Tibia": GROUND_TIBIA,
    "Leg8_Hip": -1.0,
    "Leg8_Femur": GROUND_FEMUR,
    "Leg8_Tibia": GROUND_TIBIA,
}

ZERO_POSE = {
    "Leg1_Hip": 0.0,
    "Leg1_Femur": 0.0,
    "Leg1_Tibia": 0.0,
    "Leg2_Hip": 0.0,
    "Leg2_Femur": 0.0,
    "Leg2_Tibia": 0.0,
    "Leg3_Hip": 0.0,
    "Leg3_Femur": 0.0,
    "Leg3_Tibia": 0.0,
    "Leg4_Hip": 0.0,
    "Leg4_Femur": 0.0,
    "Leg4_Tibia": 0.0,
    "Leg5_Hip": 0.0,
    "Leg5_Femur": 0.0,
    "Leg5_Tibia": 0.0,
    "Leg6_Hip": 0.0,
    "Leg6_Femur": 0.0,
    "Leg6_Tibia": 0.0,
    "Leg7_Hip": 0.0,
    "Leg7_Femur": 0.0,
    "Leg7_Tibia": 0.0,
    "Leg8_Hip": 0.0,
    "Leg8_Femur": 0.0,
    "Leg8_Tibia": 0.0,
}

PD_KP = 50.0
PD_KV = 0.5
MAX_TORQUE = 8.0


def main():
    gs.init(
        backend=gs.cpu,
        logging_level="warning",
    )

    scene = gs.Scene(
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Robot
    robot: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file=SPIDER_XML,
            pos=INITIAL_BODY_POSITION,
            quat=INITIAL_QUAT,
        ),
    )

    # Lidar sensors
    sensor = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=GridPattern(resolution=0.2, size=(0.4, 0.2)),
            entity_idx=robot.idx,
            pos_offset=(0.24, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
            draw_debug=True,
        )
    )

    # Build scene
    N_ENVS = 1
    scene.build(n_envs=N_ENVS)

    #  Init actuators
    dof_idx = [robot.get_joint(name).dof_start for name in STABLE_POS.keys()]

    # Gains and torque limits
    num_actuators = len(STABLE_POS)
    robot.set_dofs_kp([20] * num_actuators, dof_idx)
    robot.set_dofs_kv([0.5] * num_actuators, dof_idx)
    robot.set_dofs_force_range(
        [-MAX_TORQUE] * num_actuators,
        [MAX_TORQUE] * num_actuators,
        dof_idx,
    )

    # Initial dof positions
    zero_pos = torch.tensor(
        [list(ZERO_POSE.values()) for _ in range(N_ENVS)],
        device=gs.device,
    )
    stable_pos = torch.tensor(
        [list(STABLE_POS.values()) for _ in range(N_ENVS)],
        device=gs.device,
    )
    ground_pos = torch.tensor(
        [list(GROUND_POSE.values()) for _ in range(N_ENVS)],
        device=gs.device,
    )
    target_pos = stable_pos
    robot.set_dofs_position(
        position=target_pos,
        dofs_idx_local=dof_idx,
    )

    for i in range(1000):
        if i % 10 == 0:
            r = sensor.read()
            print(r.distances)

        robot.set_dofs_position(
            position=target_pos,
            dofs_idx_local=dof_idx,
        )

        scene.step()


if __name__ == "__main__":
    main()
