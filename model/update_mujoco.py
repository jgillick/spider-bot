import signal
import sys
import argparse
from os import path
import xml.etree.ElementTree as ET

from constants import (
    HIP_RANGES,
)

SOURCE_PATH = "./v2/SpiderBot.xml"


def main(tree):
    # tree = add_defaults(tree)
    tree = update_joint_values(tree)
    # tree = actuator_definitions(tree)
    tree = main_body(tree)
    tree = add_foot_friction(tree)

    # Pretty print and output
    ET.indent(tree, space="  ")
    tree.write(SOURCE_PATH, encoding="utf-8")


def update_joint_values(tree):
    """
    Update the hip joint max ranges
    """
    for n in ["R", "L"]:
        for i in range(1, 5):
            leg = f"{n}{i}"
            joint_name = f"{leg}_Hip"
            joint = tree.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                continue
            joint.set("range", " ".join(HIP_RANGES[leg]))

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

    # Motor defaults
    ET.SubElement(
        default,
        "motor",
        {
            "gear": "1",
        },
    )

    return tree


def actuator_definitions(tree):
    """
    Define the actuators
    """
    actuators = ET.SubElement(tree.getroot(), "actuator")
    for n in ["R", "L"]:
        for i in range(1, 4):
            leg = f"{n}{i}"
            for joint_name in ("Hip", "Femur", "Knee"):
                ET.SubElement(
                    actuators,
                    "motor",
                    {
                        "name": f"{leg}_{joint_name}_Actuator",
                        "joint": f"{leg}_{joint_name}",
                    },
                )

    return tree


def main_body(tree):
    """
    Add free joint and body position
    """
    root_body = tree.find("./worldbody//body[@name='model_root']")
    if root_body is None:
        print('No body named "Body" found.')
        exit(1)

    # free_joint = ET.Element(
    #     "joint",
    #     {
    #         "name": "root",
    #         "type": "free",
    #         "pos": "0 0 0",
    #         "armature": "0",
    #         "damping": "0",
    #         "limited": "false",
    #         "margin": "0.01",
    #     },
    # )
    # root_body.insert(0, free_joint)

    # Adjust position of body
    root_body.set("pos", "0.0 0.0 0.118")

    return tree


def add_foot_friction(tree):
    """
    Add friction to the feet
    """
    for n in ["R", "L"]:
        for i in range(1, 4):
            foot_name = f"{n}{i}_Foot_Foot_visual_geom"
            foot = tree.find(f".//geom[@name='{foot_name}']")
            if foot is not None:
                foot.set("friction", "2.0 0.1 0.01")
    return tree


def sigint_handler():
    sys.exit(0)


signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGINT, sigint_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update SpiderBot XML configuration")
    args = parser.parse_args()

    if not path.exists(SOURCE_PATH):
        print(f"File does not exist: {SOURCE_PATH}")
        exit(1)
    tree = ET.parse(SOURCE_PATH)
    input_dir = path.dirname(SOURCE_PATH)
    main(tree)
