import os
import genesis as gs
from genesis.engine.entities import RigidEntity

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))

INITIAL_BODY_POSITION = [0.0, 0.0, 0.14]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
FEMUR_INIT_POSITION = 0.5
TIBIA_INIT_POSITION = 0.6
INITIAL_JOINT_POSITIONS = {
    "Leg1_Hip": -1.0,
    "Leg1_Femur": FEMUR_INIT_POSITION,
    "Leg1_Tibia": TIBIA_INIT_POSITION,
    "Leg2_Hip": -1.0,
    "Leg2_Femur": FEMUR_INIT_POSITION,
    "Leg2_Tibia": TIBIA_INIT_POSITION,
    "Leg3_Hip": 1.0,
    "Leg3_Femur": FEMUR_INIT_POSITION,
    "Leg3_Tibia": TIBIA_INIT_POSITION,
    "Leg4_Hip": 1.0,
    "Leg4_Femur": FEMUR_INIT_POSITION,
    "Leg4_Tibia": TIBIA_INIT_POSITION,
    "Leg5_Hip": 1.0,
    "Leg5_Femur": FEMUR_INIT_POSITION,
    "Leg5_Tibia": TIBIA_INIT_POSITION,
    "Leg6_Hip": 1.0,
    "Leg6_Femur": FEMUR_INIT_POSITION,
    "Leg6_Tibia": TIBIA_INIT_POSITION,
    "Leg7_Hip": -1.0,
    "Leg7_Femur": FEMUR_INIT_POSITION,
    "Leg7_Tibia": TIBIA_INIT_POSITION,
    "Leg8_Hip": -1.0,
    "Leg8_Femur": FEMUR_INIT_POSITION,
    "Leg8_Tibia": TIBIA_INIT_POSITION,
}

PD_KP = 50.0
PD_KV = 0.5
MAX_TORQUE = 8.0


def get_center_above(entity: RigidEntity, height_offset=0.5):
    aabb_tensor = entity.get_AABB()
    min_coords = aabb_tensor[0]  # [min_x, min_y, min_z]
    max_coords = aabb_tensor[1]  # [max_x, max_y, max_z]

    # Calculate center coordinates
    center_x = (min_coords[0] + max_coords[0]) / 2
    center_y = (min_coords[1] + max_coords[1]) / 2
    center_z = (min_coords[2] + max_coords[2]) / 2

    # Position above the center (assuming z is vertical)
    position_above = [center_x, center_y, center_z + height_offset]

    return position_above


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Robot
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=SPIDER_XML,
            pos=INITIAL_BODY_POSITION,
            quat=INITIAL_QUAT,
        ),
    )

    # Arrow
    # arrow = scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="./robo_genesis/assets/arrow.stl",
    #         pos=[0.0, 0.0, 0.0],
    #         quat=[1.0, 0.0, 0.0, 0.0],
    #         scale=0.02,
    #         collision=False,
    #         fixed=True,
    #     ),
    #     surface=gs.surfaces.Rough(
    #         diffuse_texture=gs.textures.ColorTexture(
    #             color=(0.0, 0.5, 0.0, 1.0),
    #         ),
    #     ),
    # )

    # Build scene
    scene.build()

    # Position arrow over robot
    aabb = robot.get_AABB()
    min_aabb = aabb[0]  # [min_x, min_y, min_z]
    max_aabb = aabb[1]  # [max_x, max_y, max_z]
    center_aabb_x = (min_aabb[0] + max_aabb[0]) / 2
    center_aabb_y = (min_aabb[1] + max_aabb[1]) / 2
    arrow_pos = [center_aabb_x, center_aabb_y, max_aabb[2] + 0.05]
    # arrow.set_pos(arrow_pos)

    scene.draw_debug_arrow(
        arrow_pos, vec=(0.2, 0, 0), color=[0.0, 0.5, 0.0], radius=0.025
    )
    scene.draw_debug_arrow(
        arrow_pos, vec=(0.15, 0, 0), color=[0.0, 0.0, 0.5], radius=0.026
    )

    #  Init actuators
    dof_idx = [
        robot.get_joint(name).dof_start for name in INITIAL_JOINT_POSITIONS.keys()
    ]

    # Gains and torque limits
    num_actuators = len(INITIAL_JOINT_POSITIONS)
    robot.set_dofs_kp([PD_KP] * num_actuators, dof_idx)
    robot.set_dofs_kv([PD_KV] * num_actuators, dof_idx)
    robot.set_dofs_force_range(
        [-MAX_TORQUE] * num_actuators,
        [MAX_TORQUE] * num_actuators,
        dof_idx,
    )

    # Initial dof positions
    robot.set_dofs_position(
        position=list(INITIAL_JOINT_POSITIONS.values()),
        dofs_idx_local=dof_idx,
        zero_velocity=True,
    )

    for i in range(1000):
        if i == 20:
            robot.control_dofs_position(
                list(INITIAL_JOINT_POSITIONS.values()),
                dof_idx,
            )

        scene.step()


if __name__ == "__main__":
    main()
