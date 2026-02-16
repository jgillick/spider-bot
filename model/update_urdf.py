import re
import os
import math
import numpy as np
import argparse
import xml.etree.ElementTree as ET

from constants import (
    HIP_RANGES,
    FEMUR_RANGE,
    TIBIA_RANGE,
)


SOURCE_PATH = "./export/urdf/SpiderBody/SpiderBody.urdf"


def main(tree, output_path):
    tree = simplify_names(tree)
    tree = update_join_ranges(tree)

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
        # from name, link, and filename attributes
        for attr in ("name", "link", "filename"):
            if attr in elem.attrib:
                value = elem.attrib[attr]
                value = re.sub(r"Spider-Leg-Assembly-v\d+_", "", value)
                value = re.sub(r"GIM6010-8-v\d+_", "", value)

                # Names can not have dashes
                if attr == "name" or attr == "link":
                    value = re.sub(r"-", "_", value)

                # Update meshes directory from meshes/ to meshes/urdf/
                if attr == "filename":
                    value = re.sub(r"meshes/", "meshes/urdf/", value)

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
        joint = tree.find(f".//joint[@name='{joint_name}']/limit")
        if joint is not None:
            joint.set("lower", HIP_RANGES[i][0])
            joint.set("upper", HIP_RANGES[i][1])

        # Femur
        joint_name = f"{leg_name}_Femur"
        joint = tree.find(f".//joint[@name='{joint_name}']/limit")
        if joint is not None:
            joint.set("lower", FEMUR_RANGE[0])
            joint.set("upper", FEMUR_RANGE[1])

        # Tibia
        joint_name = f"{leg_name}_Tibia"
        joint = tree.find(f".//joint[@name='{joint_name}']/limit")
        if joint is not None:
            joint.set("lower", TIBIA_RANGE[0])
            joint.set("upper", TIBIA_RANGE[1])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update SpiderBot URDF configuration")
    parser.add_argument("output_path", help="Output file path for the generated XML")

    args = parser.parse_args()

    if not os.path.exists(SOURCE_PATH):
        print(f"File does not exist: {SOURCE_PATH}")
        exit(1)
    tree = ET.parse(SOURCE_PATH)
    main(
        tree,
        output_path=args.output_path,
    )
