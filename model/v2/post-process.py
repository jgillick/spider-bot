import signal
import sys
from os import path
import xml.etree.ElementTree as ET

SOURCE_PATH = "./SpiderBot.xml"
LIMITS = {
    "R1_Hip": (-0.3, 1.4),
    "R2_Hip": (-0.4, 0.9),
    "R3_Hip": (-0.9, 0.4),
    "R4_Hip": (-1.4, 0.3),
    "L1_Hip": (-1.4, 0.3),
    "L2_Hip": (-0.9, 0.4),
    "L3_Hip": (-0.4, 0.9),
    "L4_Hip": (-0.3, 1.4),
}


def main(tree):
    tree = joint_limits(tree)
    tree = add_foot_friction(tree)
    ET.indent(tree, space="  ")
    tree.write(SOURCE_PATH, encoding="utf-8")


def joint_limits(tree):
    """
    Add joint limits to the tree
    """
    for name, limits in LIMITS.items():
        joint = tree.find(f".//joint[@name='{name}']")
        if joint is not None:
            joint.set("range", f"{limits[0]} {limits[1]}")
    return tree


def add_foot_friction(tree):
    """
    Add friction to the feet
    """
    feet = tree.findall(f".//geom[@mesh='Foot_collision00']")
    for foot in feet:
        if foot is not None:
            foot.set("friction", "2.0 0.1 0.01")
    return tree


def sigint_handler():
    sys.exit(0)


signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGINT, sigint_handler)


if __name__ == "__main__":
    tree = ET.parse(SOURCE_PATH)
    input_dir = path.dirname(SOURCE_PATH)
    main(tree)
