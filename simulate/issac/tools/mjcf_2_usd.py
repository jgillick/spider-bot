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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIENCE_KIT_PATH = os.path.join(THIS_DIR, "isaaclab.python.mjcf.kit")

simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True, "fast_shutdown": True},
    experience=EXPERIENCE_KIT_PATH,
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
from pxr import PhysxSchema, Gf, Sdf, UsdShade, UsdPhysics, Usd

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")

# parse the arguments
args_cli = parser.parse_args()

ROOT_PATH = "/SpiderBotNoEnv"


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
            joint = stage.GetPrimAtPath(f"{ROOT_PATH}/joints/Leg{leg}_{joint_name}")

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


def ignore_leg_self_collisions(stage: Usd.Stage):
    """
    Disable collisions for bodies in the same leg.
    Bodies part of the same leg might rub against each other but should not collide, otherwise
    the leg will not move. We observed this problem mostly with the hip and femur movements.
    """
    leg_parts = [
        "Hip_actuator_assembly_Body_Bracket",
        "Hip_actuator_assembly_Motor",
        "Hip_actuator_assembly_Hip_Bracket",
        "Femur_actuator_assembly_Motor",
        "Femur_actuator_assembly_Femur",
        "Knee_actuator_assembly_Motor",
        "Tibia_Leg",
    ]
    for leg in range(1, 9):
        group_path = f"{ROOT_PATH}/CollisionGroups/Leg{leg}"
        collision_group = UsdPhysics.CollisionGroup.Define(stage, group_path)

        # Find all collision bodies for this leg
        leg_collision_prims = []
        for part_name in leg_parts:
            body = stage.GetPrimAtPath(f"{ROOT_PATH}/Body/Leg{leg}_{part_name}")
            for prim in body.GetAllChildren():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    leg_collision_prims.append(prim.GetPath())

        # Add leg parts to this collision group
        collection_api = collision_group.GetCollidersCollectionAPI()
        collection_api.CreateIncludesRel().SetTargets(leg_collision_prims)

        # Make this group not collide with itself
        collision_group.CreateFilteredGroupsRel().AddTarget(group_path)


def set_material_color(stage: Usd.Stage):
    """
    Set the default material color to black.
    """
    def_shader = UsdShade.Material.Get(
        stage, f"{ROOT_PATH}/Looks/DefaultMaterial/DefaultMaterial"
    )
    def_shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.05, 0.05, 0.05)
    )


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
        worldbody = stage.GetPrimAtPath(f"{ROOT_PATH}/worldBody")
        if worldbody:
            worldbody.SetActive(False)

        # Other fixes
        fix_joint_api(stage)
        ignore_leg_self_collisions(stage)
        set_material_color(stage)

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
