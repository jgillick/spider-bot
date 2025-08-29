from typing import Tuple, Sequence, Union

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager

CommandRangeValue = Tuple[float, float]
CommandRange = Union[CommandRangeValue, dict[str, CommandRangeValue]]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CommandManager(BaseManager):
    """
    Generates a command from uniform distribution of values.

    Example:
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def configuration_managers(self):
                self.height_command = CommandManager(self, range=(0.1, 0.2))

            def step(self, actions: torch.Tensor):
                super().step(actions)
                # ...handle actions and rewards calculations ...

                self.command_manager.step()

                return obs, rewards, terminations, timeouts, info


            def reset(self, env_ids: Sequence[int] = None):
                super().reset(env_ids)
                # ...do reset logic here...

                self.command_manager.reset(envs_ids)
                return obs, info

            def calculate_rewards():
                target_height = self.command_manager.command
                base_pos = self.robot.get_pos()
                height_reward = torch.square(base_pos[:, 2] - target_height)

                # ...additional reward calculations here...

    Debug Visualization:
        If you set `debug_visualizer` to True, arrows will be rendered above your robot
        showing the commanded velocity vs the actual velocity.
        The commanded velocity is green and the actual velocity is blue.

    Args:
        env: The environment to control
        range: The number range, or dict of ranges, to generate target command(s) for
        resample_time_s: The time interval between changing the command
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: CommandRange,
        resample_time_sec: float = 5.0,
    ):
        super().__init__(env)
        self._range = range
        self.resample_time_sec = resample_time_sec

        num_ranges = len(range) if isinstance(range, dict) else 1
        self._command = torch.zeros(env.num_envs, num_ranges, device=gs.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, num_ranges)."""
        return self._command
    
    @property
    def range(self) -> CommandRange:
        """The range of values to generate target command(s) for."""
        return self._range
    
    @range.setter
    def range(self, range: CommandRange):
        """Set the range of values to generate target command(s) for."""
        # Validate the shape of the range
        num = len(range) if isinstance(range, dict) else 1
        if num != self._command.shape[1]:
            raise ValueError(f"Cannot change the shape of the CommandManager range. Expected size: {self._command.shape[1]}, got {num}")
        # Validate the range types match
        if type(range) != type(self._range):
            raise ValueError(f"Cannot change the base type of the CommandManager range. Expected type: {type(self._range)}, got {type(range)}")
        # Validate that the dict keys match the current range dict keys
        if isinstance(range, dict):
            if set(range.keys()) != set(self._range.keys()):
                raise ValueError(f"Cannot change the dict keys of the CommandManager range. Expected keys: {set(self._range.keys())}, got {set(range.keys())}")
        self._range = range
    
    @property
    def resample_time_sec(self) -> float:
        """The time interval (in seconds) between changing the command for each environment."""
        return self._resample_time_sec
    
    @resample_time_sec.setter
    def resample_time_sec(self, resample_time_s: float):
        """Set the time interval (in seconds) between changing the command for each environment."""
        self._resample_time_sec = resample_time_s
        self._resample_steps = int(resample_time_s / self.env.dt)
    

    """
    Operations
    """

    def step(self):
        """Resample the command if necessary"""
        if not self.enabled:
            return

        resample_command_envs = (
            (self.env.episode_length % self._resample_steps == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_command(resample_command_envs)

    def reset(self, env_ids: Sequence[int] = None):
        """One or more environments have been reset"""
        if not self.enabled:
            return
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        self._resample_command(env_ids)

    """
    Implementation
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Create a new command for the given environment ids."""
        num = torch.empty(len(env_ids), device=gs.device)

        # Get range values (this might have changed since init due to curriculum training)
        ranges = None
        if isinstance(self._range, dict):
            ranges = list(self._range.values())
        else:
            ranges = [self._range]

        # Resample the command
        for i in range(self._command.shape[1]):
            self._command[env_ids, i] = num.uniform_(*ranges[i])
