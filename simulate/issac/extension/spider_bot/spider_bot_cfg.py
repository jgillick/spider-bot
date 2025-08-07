"""Configuration for the SpiderBot robot."""

import os
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

# Get the path to the SpiderBot USD file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_BOT_FILE = "SpiderBot.usd"
ASSETS_DIR = os.path.abspath(os.path.join(THIS_DIR, "assets"))
SPIDER_BOT_USD_PATH = os.path.join(ASSETS_DIR, SPIDER_BOT_FILE)


@configclass
class SpiderBotCfg(ArticulationCfg):
    """Configuration for the 8-legged spider robot"""

    spawn = sim_utils.UsdFileCfg(
        usd_path=SPIDER_BOT_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=1,
        ),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.16),
        joint_pos={
            # Hip joints
            "Leg[1-2]_Hip": -1.0,
            "Leg[3-6]_Hip": 1.0,
            "Leg[7-8]_Hip": -1.0,
            # Femur joints
            "Leg[1-8]_Femur": 0.5,
            # Tibia joints
            "Leg[1-8]_Tibia": 0.5,
        },
    )

    actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=8.0,
            velocity_limit_sim=12.0,  # Rated: 12.5 rad/s, Max: 43 rad/s
            stiffness={".*": 150.0},
            damping={".*": 0.5},
        ),
    }

    # Limit joints to 90% of their range
    soft_joint_pos_limit_factor = 0.9

    # Collision properties to apply to all rigid bodies
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.001,
        rest_offset=0.0,
    )
