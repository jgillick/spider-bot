import re
import os
import math
import numpy as np
import argparse
import xml.etree.ElementTree as ET

from constants import (
    ACTUATOR_TORQUE_RANGE,
    HIP_RANGES,
    FEMUR_RANGE,
    TIBIA_RANGE,
)

SOURCE_PATH = "./export/mujoco/SpiderBody/SpiderBody.xml"

MOTOR_MAX_TORQUE = 11.0
MOTOR_STIFFNESS = 20.0
MOTOR_DAMPING = 1.0

JOINT_AXIS = {
    "Hip": "0 1 0",
    "Femur": "1 0 0",
    "Tibia": "1 0 0",
}


def main(tree, output_path, ground=False, light=False):
    tree = simplify_names(tree)
    tree = update_joint_values(tree)

    tree = remove_default_ground_plane(tree)
    if ground:
        tree = ground_plain(tree)
    if not light:
        tree = remove_default_light(tree)

    tree = visual_settings(tree)
    tree = add_defaults(tree)
    tree = actuator_definitions(tree)
    tree = main_body(tree)
    tree = add_foot_friction(tree)
    tree = convert_euler_to_quat(tree)

    # Pretty print and output
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8")


def simplify_names(tree):
    """
    Clean up element names and references
    """
    for elem in tree.iter():
        for attr in ("name", "mesh", "file", "joint"):
            if attr in elem.attrib:
                value = elem.attrib[attr]

                # Remove "Spider-Leg-Assembly-vXYZ_" and "GIM6010-8-vXYZ"
                value = re.sub(r"Spider-Leg-Assembly-v\d+_", "", value)
                value = re.sub(r"GIM6010-8-v\d+_", "", value)

                # Remove "geom" suffix from names
                if attr == "name":
                    value = re.sub(r"_geom$", "", value)

                # Update meshes directory from meshes/ to meshes/mujoco/
                if attr == "file":
                    value = re.sub(r"meshes/", "meshes/mujoco/", value)

                # Remove dashes from names (these are not good when converting to USD)
                if attr == "name" or attr == "joint" or attr == "mesh":
                    value = re.sub(r"-", "_", value)

                # Simplify join names
                if elem.tag == "joint" and attr == "name":
                    value = re.sub(
                        r"Hip_actuator_assembly_Hip_Bracket_Hip", "Hip", value
                    )
                    value = re.sub(
                        r"Femur_actuator_assembly_Femur_Revolute_2", "Femur", value
                    )
                    value = re.sub(r"Tibia_Leg_Tibia", "Tibia", value)

                elem.attrib[attr] = value

    return tree


def update_joint_values(tree):
    """
    Update the joint max range, damping, and stiffness
    """

    # Damping and stiffness
    for leg in range(1, 9):
        for joint_type in ["Hip", "Femur", "Tibia"]:
            joint_name = f"Leg{leg}_{joint_type}"
            joint = tree.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                continue

            joint.attrib["damping"] = str(MOTOR_DAMPING)
            joint.attrib["stiffness"] = str(MOTOR_STIFFNESS)
            joint.set("axis", JOINT_AXIS[joint_type])

            # Hip range and axis
            if joint_type == "Hip":
                joint.set("range", " ".join(HIP_RANGES[leg - 1]))

            # Femur range
            if joint_type == "Femur":
                joint.set("range", " ".join(FEMUR_RANGE))

            # Tibia range
            if joint_type == "Tibia":
                joint.set("range", " ".join(TIBIA_RANGE))

    return tree


def remove_default_ground_plane(tree):
    """
    Remove the default ground plane from the XML tree.
    """
    worldbody = tree.find("./worldbody")
    if worldbody is None:
        print("No <worldbody> found.")
        exit(1)

    ground_plane = tree.find("./worldbody/geom[@type='plane']")
    if ground_plane is not None:
        worldbody.remove(ground_plane)

    return tree


def remove_default_light(tree):
    """
    Remove the default light element
    """
    worldbody = tree.find("./worldbody")
    if worldbody is None:
        print("No <worldbody> found.")
        exit(1)

    light = tree.find("./worldbody/light")
    if light is not None:
        worldbody.remove(light)

    return tree


def ground_plain(tree):
    """
    Add a better ground plane
    """
    # Ensure there are container elements
    assets = tree.find("./asset")
    worldbody = tree.find("./worldbody")
    if assets is None:
        assets = ET.SubElement(tree.getroot(), "asset")
    if worldbody is None:
        worldbody = ET.SubElement(tree.getroot(), "worldbody")

    # Create ground plane texture and material
    ET.SubElement(
        assets,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        assets,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    ground_plane = ET.Element(
        "geom",
        {
            "name": "floor",
            "size": "40 40 40",
            "type": "plane",
            "material": "groundplane",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "1 1 1 1",
        },
    )
    worldbody.insert(0, ground_plane)

    return tree


def add_defaults(tree):
    """
    Add default settings to the XML tree.
    """
    # Insert default element after compiler, if it exists
    default = tree.find("./default")
    if default is None:
        compiler = tree.find("./compiler")
        default = ET.Element("default")
        insert_at = 0
        if compiler is not None:
            insert_at = list(tree.getroot()).index(compiler) + 1
        tree.getroot().insert(insert_at, default)

    # Add geom defaults
    ET.SubElement(
        default,
        "geom",
        {"contype": "1", "conaffinity": "1", "rgba": "0.1 0.1 0.1 1"},
    )

    return tree


def visual_settings(tree):
    """
    Add visual settings to the XML tree.
    """
    # Insert visual element after compiler, if it exists
    visual = tree.find("./visual")
    if visual is None:
        compiler = tree.find("./compiler")
        visual = ET.Element("visual")
        insert_at = 0
        if compiler is not None:
            insert_at = list(tree.getroot()).index(compiler) + 1
        tree.getroot().insert(insert_at, visual)

    ET.SubElement(visual, "global", {"offwidth": "1200", "offheight": "800"})

    return tree


def actuator_definitions(tree):
    """
    Define the actuators
    """
    actuators = ET.SubElement(tree.getroot(), "actuator")
    for i in range(1, 9):
        for joint_name in ("Hip", "Femur", "Tibia"):
            ET.SubElement(
                actuators,
                "motor",
                {
                    "name": f"Leg{i}_{joint_name}_Actuator",
                    "joint": f"Leg{i}_{joint_name}",
                    "ctrlrange": " ".join(ACTUATOR_TORQUE_RANGE),
                    "forcerange": f"{-MOTOR_MAX_TORQUE} {MOTOR_MAX_TORQUE}",
                    "forcelimited": "true",
                    "gear": "1",
                },
            )

    return tree


def main_body(tree):
    """
    Add free joint, bottom plane, IMU, and a camera to the main body
    """
    root_body = tree.find("./worldbody//body[@name='Body']")
    if root_body is None:
        print('No body named "Body" found.')
        exit(1)

    # sensor = tree.find("./sensor")
    # if sensor is None:
    #     sensor = ET.SubElement(tree.getroot(), "sensor")

    free_joint = ET.Element(
        "joint",
        {
            "name": "root",
            "type": "free",
            "pos": "0 0 0",
            "armature": "0",
            "damping": "0",
            "limited": "false",
            "margin": "0.01",
        },
    )
    # imu_site = ET.Element(
    #     "site",
    #     {
    #         "name": "imu_site",
    #         "pos": "0.24115 0 0",
    #         "size": "0.01",
    #     },
    # )
    body_cam = ET.Element(
        "camera",
        {
            "name": "track",
            "mode": "targetbodycom",
            "target": "Body",
            "pos": "-0.817 -1.628 0.6",
        },
    )
    ground_cam = ET.Element(
        "camera",
        {
            "name": "groundcam",
            "mode": "targetbodycom",
            "target": "Body",
            "pos": "-1.0 0 0.1",
        },
    )
    # head = ET.Element(
    #     "geom",
    #     {
    #         "name": "head",
    #         "pos": "0.49 0 0.015",
    #         "size": "0.025",
    #         "type": "sphere",
    #         "rgba": "0 .5 0 1",
    #     },
    # )

    # Bottom planes that should cover the entire bottom of the robot
    # (this makes it easier to detect when the robot is on the ground)
    # bottom1 = ET.Element(
    #     "geom",
    #     {
    #         "name": "body_bottom1",
    #         "pos": "0.24115 0 -0.075",
    #         "size": "0.17 0.175 0.0001",
    #         "type": "box",
    #         "rgba": "1 0 0 0",
    #     },
    # )
    # bottom2 = ET.Element(
    #     "geom",
    #     {
    #         "name": "body_bottom2",
    #         "pos": "0.24115 0 -0.075",
    #         "size": "0.3 0.12 0.0001",
    #         "type": "box",
    #         "rgba": "1 0 0 0",
    #     },
    # )
    # root_body.insert(0, bottom1)
    # root_body.insert(0, bottom2)

    # root_body.insert(0, head)
    # root_body.insert(0, imu_site)
    root_body.insert(0, ground_cam)
    root_body.insert(0, body_cam)
    root_body.insert(0, free_joint)

    # Add IMU sensor
    # ET.SubElement(sensor, "gyro", {"name": "gyro_sensor", "site": "imu_site"})
    # ET.SubElement(
    #     sensor, "accelerometer", {"name": "accelerometer_sensor", "site": "imu_site"}
    # )

    # Adjust position of body
    root_body.set("pos", "0.0 0.0 0.134")

    return tree


def add_foot_friction(tree):
    """
    Add friction to the feet
    """
    for i in range(1, 9):
        foot_name = f"Leg{i}_Tibia_Foot_geom"
        foot = tree.find(f".//geom[@name='{foot_name}']")
        if foot is not None:
            foot.set("friction", "2.0 0.1 0.01")
    return tree


def multiply_quat(qa, qb):
    """
    Multiply two quaternions.
    Ported from mujoco's mju_mulQuat

    Args:
        qa: First quaternion [w, x, y, z]
        qb: Second quaternion [w, x, y, z]

    Returns:
        Result quaternion [w, x, y, z]
    """
    result = np.array(
        [
            qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
            qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
            qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
            qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
        ]
    )
    return result


def mujoco_euler_to_quat(euler, seq="XYZ"):
    """
    Convert Euler angles to quaternion using MuJoCo's logic.

    This function follows MuJoCo's mju_euler2Quat implementation:
    - For extrinsic sequences (XYZ, ZYX, etc.): applies rotations in world frame
    - For intrinsic sequences (xyz, zyx, etc.): applies rotations in body frame

    Args:
        euler: [x, y, z] angles in radians
        euler_seq: Euler sequence ('XYZ', 'xyz', 'ZYX', 'zyx', etc.)

    Returns:
        quaternion: [w, x, y, z] in MuJoCo format
    """
    try:
        tmp = np.array([1.0, 0.0, 0.0, 0.0])

        for i in range(3):
            # Construct quaternion rotation
            half_angle = euler[i] / 2
            rot = np.array([math.cos(half_angle), 0.0, 0.0, 0.0])
            sa = math.sin(half_angle)

            axis = seq[i].lower()
            if axis == "x":
                rot[1] = sa
            elif axis == "y":
                rot[2] = sa
            elif axis == "z":
                rot[3] = sa
            else:
                raise ValueError(
                    f"seq[{i}] is '{seq[i]}', should be one of x, y, z, X, Y, Z"
                )

            # Accumulate rotation
            if seq[i].islower():  # intrinsic: moving axes, post-multiply
                tmp = multiply_quat(tmp, rot)
            else:  # extrinsic: fixed axes, pre-multiply
                tmp = multiply_quat(rot, tmp)

        return tmp

    except Exception as e:
        print(f"Error converting Euler angles {euler} with sequence {seq}: {e}")
        return None


def convert_euler_to_quat(tree):
    """
    Convert all euler attributes to quat attributes in the XML tree.
    """
    # Get euler sequence from compiler
    eulerseq = "xyz"
    compiler = tree.find("./compiler")
    if compiler is not None:
        if "eulerseq" in compiler.attrib:
            eulerseq = compiler.attrib["eulerseq"]

    # Convert all euler attributes to quat attributes
    for elem in tree.iter():
        if "euler" in elem.attrib:
            euler_str = elem.attrib["euler"]
            euler = [float(x) for x in euler_str.strip().split()]

            quat = mujoco_euler_to_quat(euler, eulerseq)
            if quat is not None:
                quat_str = (
                    f"{quat[0]:.15g} {quat[1]:.15g} {quat[2]:.15g} {quat[3]:.15g}"
                )
                elem.attrib["quat"] = quat_str
                del elem.attrib["euler"]

    if compiler is not None:
        del compiler.attrib["eulerseq"]

    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update SpiderBot XML configuration")
    parser.add_argument("output_path", help="Output file path for the generated XML")
    parser.add_argument(
        "--ground", action="store_true", help="Add ground plane", default=False
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Add light",
        default=False,
    )

    args = parser.parse_args()

    if not os.path.exists(SOURCE_PATH):
        print(f"File does not exist: {SOURCE_PATH}")
        exit(1)
    tree = ET.parse(SOURCE_PATH)
    main(
        tree,
        output_path=args.output_path,
        ground=args.ground,
        light=args.light,
    )
