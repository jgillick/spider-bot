
from __future__ import annotations
from typing import TYPE_CHECKING

import sys
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from . import actions_cfg

class SpiderJointPositionAction(JointAction):
    """
    Moves the joint relative to the current joint position.
    The action input is -1.0 to 1.0 and is then mapped to the relative joint position range.
    """

    cfg: actions_cfg.SpiderJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.SpiderJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions

        # Rescale the actions from [-1, 1] to the range specified in the config
        actions = self._processed_actions.clamp(-1.0, 1.0)
        range = torch.tensor(self.cfg.range)
        actions = math_utils.unscale_transform(actions, range[0], range[1])
        self._processed_actions[:] = actions[:]

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
