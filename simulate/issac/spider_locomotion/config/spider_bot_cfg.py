"""Configuration for the SpiderBot robot."""

import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# Get the path to the SpiderBot XML file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_BOT_XML_PATH = os.path.abspath(
    os.path.join(
        "../../..",
        THIS_DIR,
        "robot",
        "SpiderBotNoGround.xml",
    )
)


@configclass
class SpiderBotCfg(ArticulationCfg):
    """Configuration for the 8-legged spider robot."""

    spawn = sim_utils.MjcfFileCfg(
        asset_path=SPIDER_BOT_XML_PATH,
        fix_base=False,
        # MuJoCo to Isaac Sim conversion settings
        import_inertia_tensor=True,
        import_sites=False,
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # Start 15cm above ground
        joint_pos={
            # Hip joints
            "Leg1_Hip": -1.0,
            "Leg2_Hip": -1.0,
            "Leg3_Hip": 1.0,
            "Leg4_Hip": 1.0,
            "Leg5_Hip": 1.0,
            "Leg6_Hip": 1.0,
            "Leg7_Hip": -1.0,
            "Leg8_Hip": -1.0,
            # Femur joints
            ".*_Femur": 0.75,
            # Tibia joints
            ".*_Tibia": 1.0,
        },
        joint_vel={".*": 0.0},
    )

    actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=8.0,  # Based on the original max_torque
            velocity_limit=10.0,
            stiffness={
                ".*_Hip": 15.0,  # Based on original position_gain
                ".*_Femur": 15.0,
                ".*_Tibia": 15.0,
            },
            damping={
                ".*_Hip": 0.8,  # Based on original velocity_gain
                ".*_Femur": 0.8,
                ".*_Tibia": 0.8,
            },
        ),
    }

    # Define collision properties for better simulation
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.01,
        rest_offset=0.0,
    )
