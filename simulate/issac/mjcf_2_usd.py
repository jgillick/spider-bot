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

import omni.kit.app
import omni.usd
import omni.kit.usd.layers

import omni.usd.commands

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.assets import check_file_path
from pxr import PhysxSchema, Gf, Sdf, UsdShade, UsdPhysics

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")

# parse the arguments
args_cli = parser.parse_args()


def printout(info_str):
    """Simple print wrapper, since the Python print() statements are suppressed sometimes."""
    sys.stdout.write(f"{info_str}\n")
    sys.stdout.flush()


def fix_joint_api(stage: Usd.Stage):
    """
    The built=in MJCF exporter uses the wrong APIs for revolute joints.
    They should be "angular" APIs (PhysicsDriveAPI:angular), but instead they are "X" APIs (PhysicsDriveAPI:X)
    """
    for leg in range(1, 9):
        for joint_name in ["Hip", "Femur", "Tibia"]:
            joint = stage.GetPrimAtPath(f"/SpiderBotNoEnv/joints/Leg{leg}_{joint_name}")

            ##
            # Drive/Limit API needs to change from "X" to "angular"

            # Enable angular APIs
            drive_api = UsdPhysics.DriveAPI.Apply(joint, "angular")
            limit_api = None
            has_limit_api = joint.HasAPI(PhysxSchema.PhysxLimitAPI, "X")
            if has_limit_api:
                limit_api = UsdPhysics.LimitAPI.Apply(joint, "angular")

            # Copy attributes from drive:X:physics.*
            attrs = joint.GetAttributes()
            for attr in attrs:
                namespace = attr.GetNamespace()
                base_name = attr.GetBaseName()

                if not namespace.startswith(
                    "drive:X:physics"
                ) and not namespace.startswith("physics"):
                    continue

                value = attr.Get()
                joint.RemoveProperty(attr.GetName())
                match base_name:
                    case "damping":
                        drive_api.CreateDampingAttr(value)
                    case "maxForce":
                        drive_api.CreateMaxForceAttr(value)
                    case "stiffness":
                        drive_api.CreateStiffnessAttr(value)
                    case "targetPosition":
                        drive_api.CreateTargetPositionAttr(value)
                    case "targetVelocity":
                        drive_api.CreateTargetVelocityAttr(value)
                    case "type":
                        drive_api.CreateTypeAttr(value)
                    case "lowerLimit":
                        if limit_api:
                            limit_api.CreateLowAttr(value)
                    case "upperLimit":
                        if limit_api:
                            limit_api.CreateHighAttr(value)

            # Remove "X" APIs
            joint.RemoveAPI(UsdPhysics.DriveAPI, "X")
            joint.RemoveAPI(PhysxSchema.PhysxLimitAPI, "X")


async def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        printout(f"ERROR: Invalid file path: {mjcf_path}")
        return False

    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)
    dest_filename = os.path.basename(dest_path)

    # Create temp directory for initial conversion
    with tempfile.TemporaryDirectory() as temp_dir_path:
        printout(f"Mujoco input: {mjcf_path}")

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

        # Load the USD file into a stage
        context = omni.usd.get_context()
        opened, error_msg = await context.open_stage_async(mjcf_converter.usd_path)
        if not opened:
            printout(
                f"Error: Could not open USD file {mjcf_converter.usd_path}: {error_msg}"
            )
            return False
        stage = context.get_stage()
        if not stage:
            printout(
                f"Error: No stage available after opening {mjcf_converter.usd_path}"
            )
            return False

        # Delete the worldbody prim
        worldbody = stage.GetPrimAtPath("/SpiderBotNoEnv/worldBody")
        if worldbody:
            worldbody.SetActive(False)

        # Fix the revolute joint APIs
        fix_joint_api(stage)

        # Set material color to black
        def_shader = UsdShade.Material.Get(
            stage, "/SpiderBotNoEnv/Looks/DefaultMaterial/DefaultMaterial"
        )
        def_shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.05, 0.05, 0.05)
        )

        # Flatten and export USD
        flattened_stage = stage.Flatten()
        if not flattened_stage:
            printout("Error: Failed to flatten the stage.")
            return False
        flattened_stage.Export(dest_path)
        printout(f"Generated USD file: {dest_path}")


if __name__ == "__main__":
    import asyncio

    printout("--------------- START mjcf_2_usd tool ---------------")
    success = asyncio.run(main())
    printout("--------------- END mjcf_2_usd tool ---------------")
    simulation_app.close()
    if not success:
        sys.exit(1)
