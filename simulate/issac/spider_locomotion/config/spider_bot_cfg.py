"""Configuration for the SpiderBot robot."""

import os
from isaaclab.utils import configclass
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

# Get the path to the SpiderBot USD file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_BOT_FILE = "SpiderBot.usd"
ASSETS_DIR = os.path.abspath(os.path.join(THIS_DIR, "../assets"))
SPIDER_BOT_USD_PATH = os.path.join(ASSETS_DIR, SPIDER_BOT_FILE)


@configclass
class SpiderBotCfg(ArticulationCfg):
    """Configuration for the 8-legged spider robot"""

    spawn = sim_utils.UsdFileCfg(
        usd_path=SPIDER_BOT_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,  # Add small damping for stability
            angular_damping=0.1,  # Add small damping for stability
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,  # Increase for better constraint resolution
        ),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.135),
        joint_pos={
            # Hip joints
            ".*Leg1_Hip": -1.0,
            ".*Leg2_Hip": -1.0,
            ".*Leg3_Hip": 1.0,
            ".*Leg4_Hip": 1.0,
            ".*Leg5_Hip": 1.0,
            ".*Leg6_Hip": 1.0,
            ".*Leg7_Hip": -1.0,
            ".*Leg8_Hip": -1.0,
            # Femur joints
            ".*_Femur": 1.0,
            # Tibia joints
            ".*_Tibia": 1.25,
        },
        joint_vel={".*": 0.0},
    )

    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*Leg[1-8]_Hip", ".*Leg[1-8]_Femur", ".*Leg[1-8]_Tibia"],
            # joint_names_expr=[".*"],
            effort_limit_sim=8.0,  # 80% of max torque (10 Nm)
            velocity_limit_sim=12.0,  # Slightly below nominal speed (12.57 rad/s)
            stiffness={".*": 35.0},
            damping={".*": 0.2},
        ),
    }

    # Limit joints to 95% of their range
    # soft_joint_pos_limit_factor = 0.95

    # Define collision properties for better simulation
    collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.01,
        rest_offset=0.0,
    )
