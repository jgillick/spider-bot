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


def printout(info_str):
    """Simple print wrapper, since the Python print() statements are suppressed sometimes."""
    sys.stdout.write(f"{info_str}\n")
    sys.stdout.flush()


def set_materials(stage):
    """Update the materials and set them to bodies"""

    # Default everything to the body material
    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=[f"{ROOT_PATH}/Body"],
        material_path=f"{ROOT_PATH}/Looks/material_body_material",
        strength="strongerThanDescendants",
        stage=stage,
    )

    # Create rubber material for feet
    # Static friction: ~0.8 - 1.5
    # Dynamic friction: ~0.5 - 1.0
    # Restitution (bounciness): ~0.0 - 0.2
    material_prim = stage.DefinePrim("{ROOT_PATH}/Looks/rubber", "Material")

    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material.CreateStaticFrictionAttr().Set(1.2)
    physx_material.CreateDynamicFrictionAttr().Set(1.0)
    physx_material.CreateRestitutionAttr().Set(0.15)
    physx_material.CreateFrictionCombineModeAttr().Set("multiply")
    physx_material.CreateRestitutionCombineModeAttr().Set("average")

    shade_material = UsdShade.Material(material_prim)
    surface_shader = UsdShade.Shader.Define(
        stage, material_prim.GetPath().AppendChild("Shader")
    )
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.5, 0.5, 0.5)
    )
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shade_material.CreateSurfaceOutput().ConnectToSource(
        surface_shader.ConnectableAPI(), "surface"
    )

    # Apply rubber to feet
    for leg in range(1, 9):
        foot = stage.GetPrimAtPath(f"{ROOT_PATH}/joints/Leg{leg}_Tibia_Foot")
        if foot:
            UsdPhysics.MaterialBindingAPI.Apply(foot)
            binding_api = UsdPhysics.MaterialBindingAPI(foot)
            binding_api.Bind(
                material_prim.GetPath(),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )


async def main():
    # check valid file path
    mjcf_path = os.path.abspath(args_cli.input)
    if not os.path.exists(mjcf_path):
        printout(f"ERROR: Invalid file path: {mjcf_path}")
        return False
    dest_path = os.path.abspath(args_cli.output)

    # create destination path
    printout(f"Mujoco input: {mjcf_path}")

    # Conversion config
    import_config = _mjcf.ImportConfig()
    import_config.set_fix_base(False)
    import_config.set_import_sites(True)
    import_config.set_import_inertia_tensor(True)
    import_config.set_make_instanceable(True)
    import_config.set_make_default_prim(True)
    import_config.set_merge_fixed_joints(True)
    import_config.set_self_collision(False)
    # import_config.set_convex_decomp(True)
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
    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=[f"{ROOT_PATH}/Body"],
        material_path=f"{ROOT_PATH}/Looks/material_body_material",
        strength="strongerThanDescendants",
        stage=stage,
    )

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
