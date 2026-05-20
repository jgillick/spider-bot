import re
import signal
import sys
from os import path
import xml.etree.ElementTree as ET

SOURCE_PATH = "./SpiderBot.xml"
COLLISION_GEOM_PREFIXES = ["Body", "Femur", "Tibia", "Foot", "Motor"]


def main(tree):
    tree = remove_collision_meshes(tree)
    tree = add_foot_friction(tree)
    tree = add_free_joint(tree)
    ET.indent(tree, space="  ")
    tree.write(SOURCE_PATH, encoding="utf-8")


def remove_collision_meshes(tree):
    """
    Remove collision bodies from the tree
    """
    parents = {c: p for p in tree.iter() for c in p}
    collision_meshes = tree.findall(f".//geom[@class='collision']")
    for mesh in collision_meshes:
        name = mesh.get("mesh")
        if not any(
            name.startswith(f"{prefix}_collision") for prefix in COLLISION_GEOM_PREFIXES
        ):
            parents[mesh].remove(mesh)
            body_meshes = tree.findall(f".//asset/geom[@name='{name}']")
            for body_mesh in body_meshes:
                parents[body_mesh].remove(body_mesh)

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


def add_free_joint(tree):
    """
    Add a free joint to the root body
    """
    root = tree.find(f".//body[@name='Body']")
    if root is not None:
        existing = root.find(f"./freejoint")
        if existing is not None:
            return tree
        free_joint = ET.Element("freejoint")
        root.insert(0, free_joint)
    return tree


def sigint_handler():
    sys.exit(0)


signal.signal(signal.SIGTERM, sigint_handler)
signal.signal(signal.SIGINT, sigint_handler)


if __name__ == "__main__":
    tree = ET.parse(SOURCE_PATH)
    input_dir = path.dirname(SOURCE_PATH)
    main(tree)
