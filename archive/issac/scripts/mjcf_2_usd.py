# Adapted from scripts/tools/convert_mjcf.py
"""
Utility to convert a MJCF into a flat USD file.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot.
For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``isaacsim.asset.importer.mjcf``) to convert
a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information
on the MJCF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_mjcf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit


INSTALLATION:
---------------
 * Place this and the isaaclab.python.mjcf.kit file inside the IsaacLab/tools directory.
 * Requires Isaac Sim & Lab to be installed, and must be run via the isaaclab.sh script.

USAGE EXAMPLE:
---------------
<ISAACLAB_ROOT>/isaaclab.sh -p <ISAACLAB_ROOT>/tools/mjcf_2_usd.py "./mujoco/robot.xml" "./usd/robot.usd"

"""

"""Launch Isaac Sim Simulator first."""
import os
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True, "fast_shutdown": True},
)

"""Rest everything follows."""

import os
import sys
import argparse

import omni
import omni.kit.commands
from isaacsim.asset.importer.mjcf import _mjcf
from pxr import UsdPhysics, PhysxSchema, UsdShade, Sdf

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")

# parse the arguments
args_cli = parser.parse_args()

ROOT_PATH = "/spider"
BODY_MATERIAL_PATH = f"{ROOT_PATH}/Looks/material_rbga"

JOINT_DRIVE_DAMPING = 0.2
JOINT_DRIVE_STIFFNESS = 15
JOINT_MAX_FORCE = 12

def printout(info_str):
    """Simple print wrapper, since the Python print() statements are suppressed sometimes."""
    sys.stdout.write(f"{info_str}\n")
    sys.stdout.flush()


def set_materials(stage):
    """Update the materials and set them to bodies"""

    ##
    # Create rubber material for feet
    # Static friction: ~0.8 - 1.5
    # Dynamic friction: ~0.5 - 1.0
    # Restitution (bounciness): ~0.0 - 0.2
    material_prim = stage.DefinePrim(f"{ROOT_PATH}/Looks/rubber", "Material")

    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material.CreateFrictionCombineModeAttr().Set("multiply")
    physx_material.CreateRestitutionCombineModeAttr().Set("average")
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr().Set(1.2)
    material_api.CreateDynamicFrictionAttr().Set(1.0)
    material_api.CreateRestitutionAttr().Set(0.15)

    rubber_shade_material = UsdShade.Material(material_prim)
    surface_shader = UsdShade.Shader.Define(
        stage, material_prim.GetPath().AppendChild("Shader")
    )
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.2, 0.2, 0.2)
    )
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rubber_shade_material.CreateSurfaceOutput().ConnectToSource(
        surface_shader.ConnectableAPI(), "surface"
    )

    # Apply rubber to feet
    for leg in range(1, 9):
        foot_prim = stage.GetPrimAtPath(f"{ROOT_PATH}/Body/Leg{leg}_Tibia_Foot")

        if foot_prim:
            material_binding_api = UsdShade.MaterialBindingAPI.Apply(foot_prim)
            material_binding_api.Bind(
                rubber_shade_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            printout(f"WARNING: Could the foot prim for Leg{leg}")

    ##
    # Apply default material to all bodies but feet
    default_material_prim = stage.GetPrimAtPath(f"{ROOT_PATH}/Looks/material_body_material")
    if default_material_prim:
        default_material = UsdShade.Material(default_material_prim)
        root_body = stage.GetPrimAtPath(f"{ROOT_PATH}/Body")
        if root_body:
            body_prims = root_body.GetChildren()
            for prim in body_prims:
                if not prim.GetName().endswith("_Tibia_Foot"):
                    material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    material_binding_api.Bind(
                        default_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )


def set_joint_gains(stage):
    """Add damping and stiffness to the existing UsdPhysicsDriveAPI of every joint"""
    for joint in stage.GetPrimAtPath(f"{ROOT_PATH}/joints").GetChildren():
        if joint.GetTypeName() == "PhysicsRevoluteJoint":
            api_path = f"{joint.GetPath()}.drive:angular"
            drive_api = UsdPhysics.DriveAPI.Get(stage, api_path)
            drive_api.CreateDampingAttr().Set(JOINT_DRIVE_DAMPING)
            drive_api.CreateStiffnessAttr().Set(JOINT_DRIVE_STIFFNESS)
            drive_api.CreateMaxForceAttr().Set(JOINT_MAX_FORCE)


async def main():
    # check valid file path
    mjcf_path = os.path.abspath(args_cli.input)
    if not os.path.exists(mjcf_path):
        printout(f"ERROR: Invalid file path: {mjcf_path}")
        return False
    dest_path = os.path.abspath(args_cli.output)
    printout(f"Mujoco input: {mjcf_path}")

    # Conversion config
    import_config = _mjcf.ImportConfig()
    import_config.set_make_default_prim(True)
    import_config.set_make_instanceable(True)
    import_config.set_merge_fixed_joints(True)
    import_config.set_convex_decomp(True)
    import_config.set_fix_base(False)
    import_config.set_import_sites(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_self_collision(False)
    import_config.set_default_drive_strength(11)
    import_config.set_visualize_collision_geoms(False)

    # import MJCF
    mjcf_interface = _mjcf.acquire_mjcf_interface()
    mjcf_interface.create_asset_mjcf(mjcf_path, ROOT_PATH, import_config)
    stage = omni.usd.get_context().get_stage()

    # Delete the worldbody prim
    worldbody = stage.GetPrimAtPath(f"{ROOT_PATH}/worldBody")
    if worldbody:
        worldbody.SetActive(False)

    # Set body material
    set_materials(stage)
    set_joint_gains(stage)

    # Save
    stage.Export(dest_path)
    printout(f"Generated USD file: {dest_path}")


if __name__ == "__main__":
    import asyncio

    printout("--------------- START mjcf_2_usd tool ---------------")
    success = asyncio.run(main())
    printout("--------------- END mjcf_2_usd tool ---------------")
    simulation_app.close()
    if not success:
        sys.exit(1)
