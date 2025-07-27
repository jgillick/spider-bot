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
    omni.kit.commands.execute('BindMaterialCommand',
                          prim_path=[f"{ROOT_PATH}/Body"],
                          material_path=f"{ROOT_PATH}/Looks/material_body_material",
                          strength='strongerThanDescendants',
                          stage=stage)

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
