import re
import torch
import genesis as gs
import numpy as np
from gymnasium import spaces
from typing import Sequence, Union, Any, Callable

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager

DofValue = Union[dict[str, Any], Any]


def ensure_dof_pattern(value: DofValue) -> dict[str, Any]:
    """
    Ensures the value is a dictionary in the form: {<joint name or regex>: <value>}.

    Example:
        >>> ensure_dof_pattern(50)
        {".*": 50}
        >>> ensure_dof_pattern({".*": 50})
        {".*": 50}
        >>> ensure_dof_pattern({"knee_joint": 50})
        {"knee_joint": 50}

    Args:
        value: The value to convert.

    Returns:
        A dictionary of DOF name pattern to value.
    """
    if isinstance(value, dict):
        return value
    return {".*": value}


class DofPositionActionManager(BaseManager):
    """
    Manages the DOF actuators and setting and resetting their position values.

    This manager assumes the action space is [-1, 1], and converts it to DOF positions within the limits of the actuators.

    Args:
        env: The environment to manage the DOF actuators for.

    Example::
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.action_manager = DofPositionActionManager(
                    self,
                    joint_names=".*",
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    pd_kp={".*": 50},
                    pd_kv={".*": 0.5},
                    max_force={".*": 8.0},
                )

            @property
            def action_space(self):
                return self.action_manager.action_space

            def build(self):
                super().build()
                self.action_manager.build()

            def step(self, actions: torch.Tensor):
                super().step(actions)

                self.action_manager.step(actions)
                self.scene.step()

                # ...handle other step actions here...

                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: Sequence[int] = None):
                super().reset(envs_idx)

                self.action_manager.reset(envs_idx)

                # ...do other reset logic here...
                return obs, info

    """

    _default_dofs_pos: torch.Tensor = None
    _dofs_pos_buffer: torch.Tensor = None
    _pos_limit_lower: torch.Tensor = None
    _pos_limit_upper: torch.Tensor = None

    def __init__(
        self,
        env: GenesisEnv,
        joint_names: Union[Sequence[str], str] = ".*",
        default_pos: DofValue = {".*": 0.0},
        pd_kp: DofValue = None,
        pd_kv: DofValue = None,
        max_force: DofValue = None,
        action_handler: Callable[[torch.Tensor], None] = None,
        quiet_action_errors: bool = False,
    ):
        super().__init__(env)
        self._has_initialized_dofs = False
        self.default_pos_cfg = ensure_dof_pattern(default_pos)
        self.pd_kp_cfg = ensure_dof_pattern(pd_kp)
        self.pd_kv_cfg = ensure_dof_pattern(pd_kv)
        self.max_force_cfg = ensure_dof_pattern(max_force)
        self.quiet_action_errors = quiet_action_errors
        self.action_handler = action_handler
        self._enabled_dof = None

        if isinstance(joint_names, str):
            self._joint_name_cfg = [joint_names]
        elif isinstance(joint_names, list):
            self._joint_name_cfg = joint_names
        else:
            raise TypeError(f"Invalid joint_names type: {type(joint_names)}")

    @property
    def dof_num(self) -> int:
        """
        Get the number of DOF that are enabled.
        """
        assert (
            self._enabled_dof is not None
        ), "DofPositionActionManager not initialized. You need to call <DofPositionActionManager>.build() in the build phase of the environment."

        return len(self._enabled_dof)

    @property
    def num_actions(self) -> int:
        """
        Alias for dof_num
        """
        return self.dof_num

    @property
    def action_space(self) -> tuple[float, float]:
        """
        If using the default action handler, the action space is [-1, 1].
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def dofs_idx(self) -> Sequence[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return list(self._enabled_dof.values())

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """Return the default DOF positions."""
        return self._default_dofs_pos

    def build(self):
        """
        Fetch the joint information from the robot and create the DOF indices.
        """
        self._enabled_dof = dict()

        for joint in self.env.robot.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            name = joint.name
            for pattern in self._joint_name_cfg:
                if re.match(pattern, name):
                    self._enabled_dof[name] = joint.dof_start
                    break

    def step(self, actions: torch.Tensor) -> None:
        """
        Convert the actions into DOF positions and set the DOF actuators.
        """
        if self.action_handler is not None:
            self.action_handler(actions)
        else:
            self._step_action_handler(actions)

    def update(self):
        """
        This should be called anytime any of the config values have changed (pd_kp_cfg, pd_kv_cfg, max_force_cfg, default_pos_cfg)
        """

        # Get position Limits and convert to shape (num_envs, limit)
        lower, upper = self.env.robot.get_dofs_limit(self.dofs_idx)
        self._pos_limit_lower = lower.unsqueeze(0).expand(self.env.num_envs, -1)
        self._pos_limit_upper = upper.unsqueeze(0).expand(self.env.num_envs, -1)

        # Set DOF values
        if self.pd_kp_cfg is not None:
            kp_values = self._get_dof_value_array(self.pd_kp_cfg)
            self.env.robot.set_dofs_kp(kp_values, self.dofs_idx)
        if self.pd_kv_cfg is not None:
            kv_values = self._get_dof_value_array(self.pd_kv_cfg)
            self.env.robot.set_dofs_kv(kv_values, self.dofs_idx)

        # Max force
        # The value can either be a single float or a tuple range
        if self.max_force_cfg is not None:
            max_force = self._get_dof_value_array(self.max_force_cfg)

            # Convert values to upper and lower arrays
            force_upper = [0.0] * self.dof_num
            force_lower = [0.0] * self.dof_num
            for i, value in enumerate(max_force):
                if isinstance(max_force[0], (list, tuple)):
                    force_lower[i] = value[0]
                    force_upper[i] = value[1]
                else:
                    force_lower[i] = -value
                    force_upper[i] = value

            self.env.robot.set_dofs_force_range(force_lower, force_upper, self.dofs_idx)

        # Assign default DOF positions to a tensor buffer for resetting
        if self.default_pos_cfg is not None:
            default_pos = self._get_dof_value_array(self.default_pos_cfg)
            self._default_dofs_pos = torch.tensor(default_pos, device=gs.device)
            self._dofs_pos_buffer = torch.zeros(
                (self.env.num_envs, self.num_actions),
                device=gs.device,
                dtype=gs.tc_float,
            )

    def reset(
        self,
        envs_idx: Sequence[int] = None,
        reset_to_default: bool = True,
        zero_dofs_velocity: bool = True,
    ):
        """Reset the DOF positions."""
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # On first reset, initialize DOF values and buffers
        if not self._has_initialized_dofs:
            self.update()
            self._has_initialized_dofs = True

        # Reset DOF positions
        if reset_to_default:
            self._dofs_pos_buffer[envs_idx] = self._default_dofs_pos
            if zero_dofs_velocity:
                self.env.robot.zero_all_dofs_velocity(envs_idx)
            self.env.robot.set_dofs_position(
                position=self._dofs_pos_buffer[envs_idx],
                dofs_idx_local=self.dofs_idx,
                envs_idx=envs_idx,
            )

    def get_dofs_position(self):
        """Return the position of the enabled DOFs."""
        return self.env.robot.get_dofs_position(self.dofs_idx)

    def get_dofs_velocity(self):
        """Return the velocity of the enabled DOFs."""
        return self.env.robot.get_dofs_velocity(self.dofs_idx)

    def _step_action_handler(self, actions: torch.Tensor):
        """Convert actions to position commands, and send them to the DOF actuators."""

        # Validate actions
        if not self.quiet_action_errors:
            if torch.isnan(actions).any():
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        actions = actions.clamp(-1.0, 1.0)
        self.actions = actions

        # Convert the action from -1 to 1, to absolute position within the actuator limits
        lower = self._pos_limit_lower
        upper = self._pos_limit_upper
        offset = (upper + lower) * 0.5
        target_positions = actions * (upper - lower) * 0.5 + offset

        # Set target positions
        self.env.robot.control_dofs_position(target_positions, self.dofs_idx)

    def _get_dof_value_array(self, values: DofValue) -> Sequence[Any]:
        """
        Given a DofValue dict, loop over the entries, and set the value to the DOF indices (from dofs_idx) that match the pattern.

        Args:
            values: The DOF value to convert (for example: `{".*": 50}`).

        Returns:
            A list of values for the DOF indices.
            For example, for 4 DOFs: [50, 50, 50, 50]
        """
        is_set = [False] * self.dof_num
        value_arr = [0.0] * self.dof_num
        for pattern, value in values.items():
            for i, name in enumerate(self._enabled_dof.keys()):
                if not is_set[i] and re.match(pattern, name):
                    is_set[i] = True
                    value_arr[i] = values[pattern]
        return value_arr
