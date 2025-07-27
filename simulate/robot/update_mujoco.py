import re
import os
import signal
import sys
import math
import shutil
import trimesh
import argparse
import coacd
import glob
from os import path
import numpy as np
import xml.etree.ElementTree as ET

from constants import (
    ACTUATOR_TORQUE_RANGE,
    HIP_RANGES,
    FEMUR_RANGE,
    TIBIA_RANGE,
)

SOURCE_PATH = "./export/mujoco/SpiderBody/SpiderBody.xml"

MOTOR_STIFFNESS = 15.0
MOTOR_DAMPING = 0.2
COLLISION_GEOM_CLASS = "collision"

JOINT_AXIS = {
    "hip": "0 1 0",
    "femur": "1 0 0",
    "tibia": "1 0 0",
}

# Skip collider geoms for these geom meshes
SKIP_COLLIDERS = [
    "Leg_KneeMotorPulley",
    "Leg_Knee_motor_bearings",
    "Leg_Hip_Bracket",
]

# Rename parts of the mesh
RENAME_PARTS = {
    "_Femur_actuator_assembly_Femur_Revolute_2": "_Femur",
    "_Femur_actuator_assembly_Femur": "_Femur",
    "_Knee_actuator_assembly_KneeMotorPulley": "_KneeMotorPulley",
    "_Knee_actuator_assembly_Knee_motor_bearings": "_Knee_motor_bearings",
    "_Knee_actuator_assembly_End_Bearing_Holder": "_End_Bearing_Holder",
    "_Hip_actuator_assembly_Hip_Bracket_Hip": "_Hip",
    "_Hip_actuator_assembly_Hip_Bracket": "_Hip_Bracket",
    "_Hip_actuator_assembly_Body_Bracket": "_Body_Bracket",
    "_Tibia_Leg_Tibia": "_Tibia",
}

DEFAULT_MATERIAL_NAME = "body_material"


def main(
    tree, input_dir, output_path, ground=False, light=False, head=False, imu=False
):
    output_dir = path.dirname(output_path)

    tree = create_materials(tree)
    tree = add_defaults(tree)
    tree = simplify_names(tree)
    tree = update_joint_values(tree)

    tree = remove_default_ground_plane(tree)
    if ground:
        tree = ground_plain(tree)
    if not light:
        tree = remove_default_light(tree)

    tree = visual_settings(tree)
    tree = actuator_definitions(tree)
    tree = main_body(tree, head=head, imu=imu)
    tree = add_foot_friction(tree)
    tree = convert_euler_to_quat(tree)
    tree = process_meshes(tree, input_dir, output_dir)

    # Pretty print and output
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8")


def normalize_name_string(name):
    """
    Cleanup name strings.
      - Remove the versioned component names
      - Convert dashes to underscores
      - Remove "geom" suffix from names
    """
    name = re.sub(r"Spider-Leg-Assembly-v\d+_", "", name)
    name = re.sub(r"GIM6010-8-v\d+_", "", name)

    # Remove "geom" suffix from names
    name = re.sub(r"_geom$", "", name)

    # Remove dashes from names (these are not good when converting to USD)
    name = re.sub(r"-", "_", name)

    # Rename map, if old is a substring of the name
    for old, new in RENAME_PARTS.items():
        name = name.replace(old, new)

    return name


def simplify_names(tree):
    """
    Clean up element names and references
    """
    for elem in tree.iter():
        for attr in ("name", "joint", "mesh"):
            if attr in elem.attrib:
                value = elem.attrib[attr]
                value = normalize_name_string(value)

                # Simplify join names
                # if elem.tag == "joint" and attr == "name":
                #     value = re.sub(
                #         r"Hip_actuator_assembly_Hip_Bracket_Hip", "Hip", value
                #     )
                #     value = re.sub(
                #         r"Femur_actuator_assembly_Femur_Revolute_2", "Femur", value
                #     )
                #     value = re.sub(r"Tibia_Leg_Tibia", "Tibia", value)

                elem.attrib[attr] = value

    return tree


def update_joint_values(tree):
    """
    Update the joint max range, damping, and stiffness
    """
    default = tree.find("./default")
    if default is None:
        print("No <default> found.")
        exit(1)

    # Damping and stiffness
    ET.SubElement(
        default,
        "joint",
        {
            "damping": str(MOTOR_DAMPING),
            "stiffness": str(MOTOR_STIFFNESS),
            "limited": "true",
        },
    )

    # Ranges
    hip_joint_def = ET.SubElement(
        default,
        "default",
        {"class": "hip_joint"},
    )
    femur_joint_def = ET.SubElement(
        default,
        "default",
        {"class": "femur_joint"},
    )
    tibia_joint_def = ET.SubElement(
        default,
        "default",
        {"class": "tibia_joint"},
    )
    ET.SubElement(
        hip_joint_def,
        "joint",
        {
            "axis": JOINT_AXIS["hip"],
        },
    )
    ET.SubElement(
        femur_joint_def,
        "joint",
        {
            "axis": JOINT_AXIS["femur"],
            "range": " ".join(FEMUR_RANGE),
        },
    )
    ET.SubElement(
        tibia_joint_def,
        "joint",
        {
            "axis": JOINT_AXIS["tibia"],
            "range": " ".join(TIBIA_RANGE),
        },
    )

    # Add classes to joints
    for leg in range(1, 9):
        for joint_type in ["Hip", "Femur", "Tibia"]:
            joint_name = f"Leg{leg}_{joint_type}"
            joint = tree.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                continue
            joint_class = f"{joint_type}_joint".lower()
            joint.set("class", joint_class)
            del joint.attrib["axis"]
            del joint.attrib["limited"]

            if joint_type == "Hip":
                joint.set("range", " ".join(HIP_RANGES[leg - 1]))
            elif "axis" in joint.attrib:
                del joint.attrib["axis"]

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
            "contype": "1",
            "conaffinity": "1",
            "rgba": "1 1 1 1",
            "material": "groundplane",
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

    # Mesh defaults
    ET.SubElement(
        default,
        "mesh",
        {
            "scale": "0.001 0.001 0.001",
        },
    )

    # Motor defaults
    ET.SubElement(
        default,
        "motor",
        {
            "ctrlrange": " ".join(ACTUATOR_TORQUE_RANGE),
            "forcerange": " ".join(ACTUATOR_TORQUE_RANGE),
            "forcelimited": "true",
            "gear": "1",
        },
    )

    # Add geom defaults
    ET.SubElement(
        default,
        "geom",
        {"contype": "0", "conaffinity": "0", "material": DEFAULT_MATERIAL_NAME},
    )

    # Add collision default
    collision_def = ET.SubElement(
        default,
        "default",
        {"class": COLLISION_GEOM_CLASS},
    )
    ET.SubElement(
        collision_def,
        "geom",
        {
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
            "material": "",
        },
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
                },
            )

    return tree


def main_body(tree, head=False, imu=False):
    """
    Add free joint, bottom plane, IMU, and a camera to the main body
    """
    root_body = tree.find("./worldbody//body[@name='Body']")
    if root_body is None:
        print('No body named "Body" found.')
        exit(1)

    sensor = tree.find("./sensor")
    if sensor is None and imu:
        sensor = ET.SubElement(tree.getroot(), "sensor")

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

    if head:
        head = ET.Element(
            "geom",
            {
                "name": "head",
                "pos": "0.49 0 0.015",
                "size": "0.025",
                "type": "sphere",
                "rgba": "0.0 0.5 0.9 1.9",
                "material": "",
            },
        )
        root_body.insert(0, head)
    if imu:
        imu_site = ET.Element(
            "site",
            {
                "name": "imu_site",
                "pos": "0.24115 0 0",
                "size": "0.01",
            },
        )
        root_body.insert(0, imu_site)
        ET.SubElement(sensor, "gyro", {"name": "gyro_sensor", "site": "imu_site"})
        ET.SubElement(
            sensor,
            "accelerometer",
            {"name": "accelerometer_sensor", "site": "imu_site"},
        )

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

    root_body.insert(0, ground_cam)
    root_body.insert(0, body_cam)
    root_body.insert(0, free_joint)

    # Adjust position of body
    root_body.set("pos", "0.0 0.0 0.134")

    # Adjust mass of body
    root_body_inertial = tree.find("./worldbody//body[@name='Body']/inertial")
    if root_body is None:
        print('No root body inertial" found.')
        exit(1)
    else:
        root_body_inertial.set("mass", "0.75")

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


def create_materials(tree):
    """
    Create materials for the robot
    """
    assets = tree.find("./asset")
    if not assets:
        print(f"Missing <asset> element")
        exit(1)

    ET.SubElement(
        assets,
        "material",
        {
            "name": DEFAULT_MATERIAL_NAME,
            "rgba": "0.1 0.1 0.1 1.0",
            "reflectance": "0.3",
            "shininess": "1.0",
            "roughness": "1.0",
        },
    )
    return tree


def create_collision_meshes(mesh_path, collision_dir):
    """
    Create collision meshes from a mesh file.
    """
    mesh_basename, _ = path.splitext(path.basename(mesh_path))
    out_dir = path.join(collision_dir, mesh_basename)

    def get_names(i):
        name = f"{mesh_basename}_collision{i:02d}"
        filename = path.join(out_dir, f"{i:02d}.stl")
        return name, filename

    # Skip creating a collider for this mesh
    if mesh_basename in SKIP_COLLIDERS:
        return {}

    # If the collision directory is newer than the mesh file, we don't need to regenerate
    if path.exists(out_dir) and path.getmtime(mesh_path) < path.getmtime(out_dir):
        existing = glob.glob(path.join(out_dir, "*.stl"))
        if len(existing) > 0:
            results = {}
            for i, filepath in enumerate(existing):
                name, _ = get_names(i)
                results[name] = filepath
            return results

    # Create directory if needed
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    print(f" - Creating collision meshes for {mesh_basename}")
    mesh = trimesh.load(path.abspath(mesh_path), force="mesh")
    coacd.set_log_level("error")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=0.15,
        mcts_iterations=100,
    )

    results = {}
    for i, part in enumerate(parts):
        vertices = part[0]
        faces = part[1]

        # Create trimesh object for this part
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        name, part_filename = get_names(i)
        part_mesh.export(part_filename)
        results[name] = part_filename

    return results


def process_meshes(tree, input_dir, output_dir):
    """
    Move referenced mesh files and consolodate all leg meshes, to reduce duplicate files.
    """
    print("Processing meshes (this might take a few minutes)...")

    # Get all parents in the XML tree
    parents = {c: p for p in tree.iter() for c in p}

    # Delete and recreate the mesh directory
    mesh_dir = path.join(output_dir, "meshes/mujoco/")
    if not path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    collision_dir = path.join(mesh_dir, "colliders/")
    if not path.exists(collision_dir):
        os.makedirs(collision_dir)

    assets = tree.find("./asset")
    if not assets:
        print(f"Missing <asset> element")
        exit(1)

    # Process all meshes
    meshes = tree.findall("./asset/mesh")
    existing_meshes = []
    collision_geoms = {}
    for mesh in meshes:
        # Skip if it's not a file mesh
        if "file" not in mesh.attrib:
            continue

        org_name = mesh.attrib["name"]
        orig_file = mesh.attrib["file"]
        orig_filename = path.basename(orig_file)
        orig_filepath = path.join(input_dir, orig_file)

        new_name = org_name
        new_filename = orig_filename

        # De-dupe motor meshes
        if org_name.endswith("_Motor"):
            new_name = "Motor"
            new_filename = "Motor.stl"

        # Replace "Leg[1-8]_" with "Leg_"
        new_name = re.sub(r"Leg\d+_", "Leg_", new_name)
        new_filename = re.sub(r"Leg\d+_", "Leg_", new_filename)
        new_filename = normalize_name_string(new_filename)

        # Update the attributes
        new_filepath = path.join(mesh_dir, new_filename)
        mesh.attrib["file"] = new_filepath
        mesh.attrib["name"] = new_name

        # If we haven't processed this mesh before, copy it to the output directory and setup the collision meshes
        if not new_name in existing_meshes:
            shutil.copy2(orig_filepath, new_filepath)

            # Create collision meshes
            collision_paths = create_collision_meshes(new_filepath, collision_dir)
            collision_geoms[new_name] = []
            for name, filepath in collision_paths.items():
                # Add mesh to assets
                ET.SubElement(
                    assets,
                    "mesh",
                    {
                        "name": name,
                        "file": filepath,
                    },
                )

                # Creat geom element for this collision mesh
                geom = ET.Element(
                    "geom",
                    {
                        "mesh": name,
                        "type": "mesh",
                        "pos": "0 0 0",
                        "class": "collision",
                    },
                )
                collision_geoms[new_name].append(geom)
        else:
            assets.remove(mesh)
        existing_meshes.append(new_name)

        # Update all references to the new mesh name
        geoms = tree.findall(f".//geom[@mesh='{org_name}']")
        for geom in geoms:
            if geom.attrib["mesh"] == org_name:
                geom.attrib["mesh"] = new_name

            # Add collision geoms to the parent body
            parent_body = parents[geom]
            if parent_body is not None:
                for c_geom in collision_geoms[new_name]:
                    parent_body.append(c_geom)
            else:
                print(f"No parent body found for {org_name}")

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


def sigint_handler():
    sys.exit(0)


signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGINT, sigint_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update SpiderBot XML configuration")
    parser.add_argument("output_path", help="Output file path for the generated XML")
    parser.add_argument(
        "--ground", action="store_true", help="Add ground plane", default=False
    )
    parser.add_argument(
        "--head", action="store_true", help="Add a head to the robot", default=False
    )
    parser.add_argument(
        "--imu", action="store_true", help="Add an IMU spot & sensors", default=False
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Add light",
        default=False,
    )

    args = parser.parse_args()

    if not path.exists(SOURCE_PATH):
        print(f"File does not exist: {SOURCE_PATH}")
        exit(1)
    tree = ET.parse(SOURCE_PATH)
    input_dir = path.dirname(SOURCE_PATH)
    main(
        tree,
        input_dir=input_dir,
        output_path=args.output_path,
        ground=args.ground,
        light=args.light,
        head=args.head,
        imu=args.imu,
    )
