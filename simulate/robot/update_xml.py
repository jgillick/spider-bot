import re
import os
import argparse
import xml.etree.ElementTree as ET

SOURCE_PATH = "./export/SpiderBody/SpiderBody.xml"
ACTUATOR_TORQUE_RANGE = "-10 10"

HIP_RANGES = (
    "-1.8 0",  # Leg1
    "-1.8 0",  # Leg2
    "0 1.8",  # Leg3
    "0 1.8",  # Leg4
    "0 1.8",  # Leg5
    "0 1.8",  # Leg6
    "-1.8 0",  # Leg7
    "-1.8 0",  # Leg8
)
FEMUR_RANGE = "0 1.0"
TIBIA_RANGE = "0.4 2.7"


def main(tree, output_path, ground=False, light=False):
    tree = simplify_names(tree)
    tree = update_join_ranges(tree)

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

    # Pretty print and output
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8")


def simplify_names(tree):
    """
    Remove unnecessary parts of element names in the XML tree.
    """

    # Remove "Spider-Leg-Assembly-vXYZ" from name attributes
    for elem in tree.iter():

        # Remove "Spider-Leg-Assembly-vXYZ_" and "GIM6010-8-vXYZ"
        # from name, mesh, and file attributes
        for attr in ("name", "mesh", "file"):
            if attr in elem.attrib:
                value = elem.attrib[attr]
                value = re.sub(r"Spider-Leg-Assembly-v\d+_", "", value)
                value = re.sub(r"GIM6010-8-v\d+_", "", value)
                elem.attrib[attr] = value

        # Simplify join names
        if elem.tag == "joint" and "name" in elem.attrib:
            name = elem.attrib["name"]
            name = re.sub(r"Hip-actuator-assembly_Hip-Bracket_Hip", "Hip", name)
            name = re.sub(r"Femur-actuator-assembly_Femur_Revolute-2", "Femur", name)
            name = re.sub(r"Tibia_Leg_Tibia", "Tibia", name)
            elem.attrib["name"] = name

    return tree


def update_join_ranges(tree):
    """
    Update the position range the leg joints
    """
    for i in range(0, 8):
        leg_name = f"Leg{i+1}"

        # Hip
        joint_name = f"{leg_name}_Hip"
        joint = tree.find(f".//joint[@name='{joint_name}']")
        if joint is not None:
            joint.set("range", HIP_RANGES[i])

        # Femur
        joint_name = f"{leg_name}_Femur"
        joint = tree.find(f".//joint[@name='{joint_name}']")
        if joint is not None:
            joint.set("range", FEMUR_RANGE)

        # Tibia
        joint_name = f"{leg_name}_Tibia"
        joint = tree.find(f".//joint[@name='{joint_name}']")
        if joint is not None:
            joint.set("range", TIBIA_RANGE)

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
    ET.SubElement(default, "geom", {"contype": "1", "conaffinity": "1"})

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
                    "ctrlrange": ACTUATOR_TORQUE_RANGE,
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

    sensor = tree.find("./sensor")
    if sensor is None:
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
    imu_site = ET.Element(
        "site",
        {
            "name": "imu_site",
            "pos": "0.24115 0 0",
            "size": "0.01",
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
    head = ET.Element(
        "geom",
        {
            "name": "head",
            "pos": "0.49 0 0.015",
            "size": "0.025",
            "type": "sphere",
            "rgba": "0 1 0 1",
        },
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

    root_body.insert(0, head)
    root_body.insert(0, imu_site)
    root_body.insert(0, ground_cam)
    root_body.insert(0, body_cam)
    root_body.insert(0, free_joint)

    # Add IMU sensor
    ET.SubElement(sensor, "gyro", {"name": "gyro_sensor", "site": "imu_site"})
    ET.SubElement(
        sensor, "accelerometer", {"name": "accelerometer_sensor", "site": "imu_site"}
    )

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
