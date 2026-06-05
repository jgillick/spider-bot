import os
import torch
import genesis as gs

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../model/v2/SpiderBot.xml"))

PD_KP = 30.0
PD_KV = 2.0
DAMPING = 0.04
ARMATURE = 0.0020
FRICTION_LOSS = 0.1
MAX_TORQUE = 22.0

INITIAL_BODY_POSITION = [0.0, 0.0, 0.118]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
STABLE_FEMUR = -0.6
STABLE_KNEE = 0.5

STABLE_POS = {
    "R1_Hip": 0.9,
    "R1_Femur": STABLE_FEMUR,
    "R1_Knee": STABLE_KNEE,
    "R2_Hip": 0.2,
    "R2_Femur": STABLE_FEMUR,
    "R2_Knee": STABLE_KNEE,
    "R3_Hip": -0.2,
    "R3_Femur": STABLE_FEMUR,
    "R3_Knee": STABLE_KNEE,
    "R4_Hip": -0.9,
    "R4_Femur": STABLE_FEMUR,
    "R4_Knee": STABLE_KNEE,
    "L1_Hip": -0.9,
    "L1_Femur": STABLE_FEMUR,
    "L1_Knee": STABLE_KNEE,
    "L2_Hip": -0.2,
    "L2_Femur": STABLE_FEMUR,
    "L2_Knee": STABLE_KNEE,
    "L3_Hip": 0.2,
    "L3_Femur": STABLE_FEMUR,
    "L3_Knee": STABLE_KNEE,
    "L4_Hip": 0.9,
    "L4_Femur": STABLE_FEMUR,
    "L4_Knee": STABLE_KNEE,
}

GROUND_FEMUR = 0.01
GROUND_KNEE = 2.6
GROUND_POSE = {
    "R1_Hip": 0.9,
    "R1_Femur": GROUND_FEMUR,
    "R1_Knee": GROUND_KNEE,
    "R2_Hip": 0.2,
    "R2_Femur": GROUND_FEMUR,
    "R2_Knee": GROUND_KNEE,
    "R3_Hip": -0.2,
    "R3_Femur": GROUND_FEMUR,
    "R3_Knee": GROUND_KNEE,
    "R4_Hip": -0.9,
    "R4_Femur": GROUND_FEMUR,
    "R4_Knee": GROUND_KNEE,
    "L1_Hip": -0.9,
    "L1_Femur": GROUND_FEMUR,
    "L1_Knee": GROUND_KNEE,
    "L2_Hip": -0.2,
    "L2_Femur": GROUND_FEMUR,
    "L2_Knee": GROUND_KNEE,
    "L3_Hip": 0.2,
    "L3_Femur": GROUND_FEMUR,
    "L3_Knee": GROUND_KNEE,
    "L4_Hip": 0.9,
    "L4_Femur": GROUND_FEMUR,
    "L4_Knee": GROUND_KNEE,
}

ZERO_POSE = {
    "R1_Hip": 0.0,
    "R1_Femur": 0.0,
    "R1_Knee": 0.0,
    "R2_Hip": 0.0,
    "R2_Femur": 0.0,
    "R2_Knee": 0.0,
    "R3_Hip": 0.0,
    "R3_Femur": 0.0,
    "R3_Knee": 0.0,
    "R4_Hip": 0.0,
    "R4_Femur": 0.0,
    "R4_Knee": 0.0,
    "L1_Hip": 0.0,
    "L1_Femur": 0.0,
    "L1_Knee": 0.0,
    "L2_Hip": 0.0,
    "L2_Femur": 0.0,
    "L2_Knee": 0.0,
    "L3_Hip": 0.0,
    "L3_Femur": 0.0,
    "L3_Knee": 0.0,
    "L4_Hip": 0.0,
    "L4_Femur": 0.0,
    "L4_Knee": 0.0,
}


def main():
    gs.init(
        # backend=gs.cpu,
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

    # Build scene
    N_ENVS = 1
    scene.build(n_envs=N_ENVS)

    #  Init actuators
    dof_idx = [robot.get_joint(name).dof_start for name in STABLE_POS.keys()]

    # Gains and torque limits
    num_actuators = len(STABLE_POS)
    robot.set_dofs_kp([PD_KP] * num_actuators, dof_idx)
    robot.set_dofs_kv([PD_KV] * num_actuators, dof_idx)
    robot.set_dofs_damping([DAMPING] * num_actuators, dof_idx)
    robot.set_dofs_armature([ARMATURE] * num_actuators, dof_idx)
    robot.set_dofs_frictionloss([FRICTION_LOSS] * num_actuators, dof_idx)
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
        position=ground_pos,
        dofs_idx_local=dof_idx,
    )

    for i in range(100):
        robot.control_dofs_position(
            position=ground_pos,
            dofs_idx_local=dof_idx,
        )
        scene.step()
    
    while True:
        for i in range(300):
            robot.control_dofs_position(
                position=target_pos,
                dofs_idx_local=dof_idx,
            )
            scene.step()
        for i in range(300):
            robot.control_dofs_position(
                position=ground_pos,
                dofs_idx_local=dof_idx,
            )
            scene.step()


if __name__ == "__main__":
    main()
