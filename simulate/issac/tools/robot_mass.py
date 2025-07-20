import os
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True, "fast_shutdown": True}
)

import sys
import asyncio
import argparse
import omni
from pxr import Usd, UsdPhysics

parser = argparse.ArgumentParser(
    description="Utility to convert determine the total mass of a robot."
)
parser.add_argument("model", type=str, help="The path to the input model USD file.")
args_cli = parser.parse_args()

def printout(info_str):
    """Simple print wrapper, since the Python print() statements are suppressed sometimes."""
    sys.stdout.write(f"{info_str}\n")
    sys.stdout.flush()

async def main(model_path):
    context = omni.usd.get_context()
    opened, error_msg = await context.open_stage_async(model_path)
    if not opened:
        printout(
            f"Error: Could not open USD file {model_path}: {error_msg}"
        )
        return False
    stage = context.get_stage()
    if not stage:
        printout(
            f"Error: No stage available after opening {model_path}"
        )
        return False

    total_mass = 0
    curr_prim = stage.GetPrimAtPath("/")
    for prim in Usd.PrimRange(curr_prim):
        try:
            MassAPI = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            mass = MassAPI.GetMassAttr().Get()
            if mass:
                total_mass += mass
        except Exception as e:
            printout(f"Error: {e}")
        
    printout(f"Total mass: {round(total_mass, 2)}kg")
    return True

if __name__ == "__main__":
    if args_cli.model:
        printout("--------------- START robot_mass tool ---------------")
        success = asyncio.run(main(args_cli.model))
        printout("--------------- END robot_mass tool ---------------")
        simulation_app.close()
        if not success:
            sys.exit(1)
    else:
        parser.print_usage()
        sys.exit(1)