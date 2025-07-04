import re
import os
import xml.etree.ElementTree as ET

SOURCE_PATH = "./export/SpiderBody/SpiderBody.xml"
OUT_PATH = "./SpiderBot.xml"
ACTUATOR_TORQUE_RANGE = "-10 10"


def main(tree):
    tree = simplify_names(tree)
    tree = update_hip_join_ranges(tree)
    tree = ground_plain(tree)
    tree = visual_settings(tree)
    tree = add_defaults(tree)
    tree = actuator_definitions(tree)
    tree = main_body(tree)
    tree = add_feet_and_sensors(tree)

    # Pretty print and output
    ET.indent(tree, space="  ")
    tree.write(OUT_PATH, encoding="utf-8")


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
            name = re.sub(r"Tibia_Tibia-Full_Tibia", "Tibia", name)
            elem.attrib["name"] = name

    return tree


def update_hip_join_ranges(tree):
    """
    Update the ranges of the hip joints in the XML tree.
    """
    hip_range = (
        "",
        "-0.3 1.570796",  # Leg1
        "-0.7 1.2",  # Leg2
        "-1.2 0.6",  # Leg3
        "-1.570796 0.3",  # Leg4
        "-1.570796 0.3",  # Leg5
        "-1.2 0.7",  # Leg6
        "-0.6 1.2",  # Leg7
        "-0.3 1.570796",  # Leg8
    )
    for i in range(1, 9):
        joint_name = f"Leg{i}_Hip"
        joint = tree.find(f".//joint[@name='{joint_name}']")
        if joint is not None:
            joint.set("range", hip_range[i])

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

    # Remove existing ground plane if it exists
    ground_plane = tree.find("./worldbody/geom[@type='plane']")
    if ground_plane is not None:
        worldbody.remove(ground_plane)

    # Add new ground plane
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

    ET.SubElement(visual, "global", {"offwidth": "800", "offheight": "600"})

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
    Add free joint, IMU, and a camera to the main body
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
    camera = ET.Element(
        "camera",
        {
            "name": "bodycam",
            "mode": "targetbodycom",
            "target": "Body",
            "pos": "-0.817 -1.628 0.4",
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

    root_body.insert(0, head)
    root_body.insert(0, imu_site)
    root_body.insert(0, camera)
    root_body.insert(0, free_joint)

    # Add IMU sensor
    ET.SubElement(sensor, "gyro", {"name": "gyro_sensor", "site": "imu_site"})
    ET.SubElement(
        sensor, "accelerometer", {"name": "accelerometer_sensor", "site": "imu_site"}
    )

    # Adjust position of body
    root_body.set("pos", "0.0 0.0 0.134")

    return tree


def add_feet_and_sensors(tree):
    """
    Add feet and touch sensors to each leg
    """
    sensor = tree.find("./sensor")
    if sensor is None:
        sensor = ET.SubElement(tree.getroot(), "sensor")

    for i in range(1, 9):
        tibia = tree.find(f".//worldbody//body[@name='Leg{i}_Tibia_Tibia-Full']")
        if tibia is None:
            print(f"No Leg{i}_Tibia_Tibia-Full body found.")
            exit(1)

        #  Add touch sites
        bad1 = {
            "name": f"Leg{i}_Tibia_bad_touch1_site",
            "pos": "0.04 0.048 0.148",
            "size": "0.005",
            "type": "sphere",
            "rgba": "1 0 0 1",
        }
        ET.SubElement(tibia, "site", bad1)
        ET.SubElement(tibia, "geom", bad1)

        bad2 = {
            "name": f"Leg{i}_Tibia_bad_touch2_site",
            "pos": "0.04 -0.05 0.195",
            "size": "0.005",
            "type": "sphere",
            "rgba": "1 0 0 1",
        }
        ET.SubElement(tibia, "site", bad2)
        ET.SubElement(tibia, "geom", bad2)

        foot = {
            "name": f"Leg{i}_Tibia_foot_site",
            "pos": "0.04 -0.1195 0.2018",
            "size": "0.007 0.005",
            "type": "cylinder",
            "rgba": "0 1 0 1",
            "euler": "0 1.5708 0",
        }
        ET.SubElement(tibia, "site", foot)
        ET.SubElement(tibia, "geom", foot | {"friction": "2.0 0.1 0.01"})

        # Add sensors
        ET.SubElement(
            sensor,
            "touch",
            {
                "name": f"Leg{i}_Tibia_bad_touch1",
                "site": f"Leg{i}_Tibia_bad_touch1_site",
            },
        )
        ET.SubElement(
            sensor,
            "touch",
            {
                "name": f"Leg{i}_Tibia_bad_touch2",
                "site": f"Leg{i}_Tibia_bad_touch2_site",
            },
        )
        ET.SubElement(
            sensor,
            "touch",
            {"name": f"Leg{i}_Tibia_foot", "site": f"Leg{i}_Tibia_foot_site"},
        )

    return tree


if __name__ == "__main__":
    if not os.path.exists(SOURCE_PATH):
        print(f"File does not exist: {SOURCE_PATH}")
        exit(1)
    tree = ET.parse(SOURCE_PATH)
    main(tree)
