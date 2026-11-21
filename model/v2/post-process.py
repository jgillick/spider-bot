import re
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

# Collision bitmasks. Two geoms collide when (A.contype & B.conaffinity) | (B.contype & A.conaffinity) is non-zero.
# The ground uses 0xFFFF, so all bits must stay within the lower 16.
TRUNK_BIT = 1 << 1
LEG_BITS = {
    leg: 1 << (2 + i)
    for i, leg in enumerate(["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"])
}
ALL_LEG_BITS = sum(LEG_BITS.values())


def main(tree):
    tree = joint_limits(tree)
    tree = add_foot_friction(tree)
    tree = collision_groups(tree)
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


def collision_groups(tree):
    """
    Set collision bitmasks so that:
      - Parts within the same leg never collide with each other (they slide against each other at the joints)
      - Legs collide with other legs and the ground
      - Trunk parts (everything rigidly attached to the Body) only collide with the ground
    """
    body = tree.find(".//body[@name='Body']")

    def visit(node, leg):
        joint = node.find("joint")
        if joint is not None and joint.get("type") == "hinge":
            match = re.match(r"([RL]\d)_", joint.get("name"))
            if match:
                leg = match.group(1)

        for geom in node.findall("geom"):
            if geom.get("class") != "collision":
                continue
            if leg is None:
                contype, conaffinity = TRUNK_BIT, 0
            else:
                contype, conaffinity = LEG_BITS[leg], ALL_LEG_BITS & ~LEG_BITS[leg]
            geom.set("contype", str(contype))
            geom.set("conaffinity", str(conaffinity))

        for child in node.findall("body"):
            visit(child, leg)

    visit(body, None)
    return tree


def sigint_handler():
    sys.exit(0)


signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGINT, sigint_handler)


if __name__ == "__main__":
    tree = ET.parse(SOURCE_PATH)
    input_dir = path.dirname(SOURCE_PATH)
    main(tree)
