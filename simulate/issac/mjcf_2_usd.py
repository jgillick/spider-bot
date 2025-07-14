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

"""

"""Launch Isaac Sim Simulator first."""
import os
from isaacsim import SimulationApp

kit_app_exp_path = os.environ["EXP_PATH"]
isaaclab_app_exp_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[".."] * 2, "apps"
)
experience = os.path.join(isaaclab_app_exp_path, "isaaclab.python.mjcf.kit")

simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True, "fast_shutdown": True},
    experience=experience,
)

"""Rest everything follows."""

import sys
import tempfile
import argparse

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
import omni.usd
import omni.kit.usd.layers

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")

# parse the arguments
args_cli = parser.parse_args()


async def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        print(f"Invalid file path: {mjcf_path}")
        return False

    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)
    dest_filename = os.path.basename(dest_path)

    # Create temp directory for initial conversion
    with tempfile.TemporaryDirectory() as temp_dir_path:

        # Convert USD files
        mjcf_converter_cfg = MjcfConverterCfg(
            asset_path=mjcf_path,
            usd_dir=temp_dir_path,
            usd_file_name=dest_filename,
            fix_base=False,
            import_sites=False,
            force_usd_conversion=True,
            make_instanceable=True,
        )
        mjcf_converter = MjcfConverter(mjcf_converter_cfg)

        # Flatten USD into a single file
        context = omni.usd.get_context()
        opened, error_msg = await context.open_stage_async(mjcf_converter.usd_path)
        if not opened:
            print(
                f"Error: Could not open USD file {mjcf_converter.usd_path}: {error_msg}"
            )
            return False
        stage = context.get_stage()
        if not stage:
            print(f"Error: No stage available after opening {mjcf_converter.usd_path}")
            return False
        flattened_stage = stage.Flatten()
        if not flattened_stage:
            print("Error: Failed to flatten the stage.")
            return False
        flattened_stage.Export(dest_path)

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input MJCF file: {mjcf_path}")
        print("MJCF importer config:")
        print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create mjcf converter and import the file
        # print output
        print("MJCF importer output:")
        print(f"Generated USD file: {dest_path}")
        print("-" * 80)
        print("-" * 80)


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(main())
    simulation_app.close()
    if not success:
        sys.exit(1)
