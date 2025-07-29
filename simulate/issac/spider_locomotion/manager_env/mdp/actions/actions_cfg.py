from typing import Tuple
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import  JointActionCfg

from . import actions

@configclass
class SpiderJointPositionActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = actions.SpiderJointPositionAction
    
    # The relative range of the joint movement (rad/s)
    range: Tuple[float, float] = (-1.0, 1.0)