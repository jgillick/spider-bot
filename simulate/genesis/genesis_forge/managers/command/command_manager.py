from typing import Tuple, Sequence, Union

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager

CommandRange = Tuple[float, float]

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
        range: The number range, or list of ranges, to generate target values for
        resample_time_s: The time interval between changing the command
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: Union[CommandRange, dict[str, CommandRange]],
        resample_time_s: float = 5.0,
    ):
        super().__init__(env)
        self.range = range
        num_ranges = len(range) if isinstance(range, dict) else 1
        self._command = torch.zeros(env.num_envs, num_ranges, device=gs.device)
        self._resample_steps = int(resample_time_s / env.dt)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self._command

    def step(self):
        """Resample the command if necessary"""
        resample_command_envs = (
            (self.env.episode_length % self._resample_steps == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_command(resample_command_envs)

    def reset(self, env_ids: Sequence[int] = None):
        """One or more environments have been reset"""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        """Create a new command for the given environment ids."""
        num = torch.empty(len(env_ids), device=gs.device)

        # Get range values (this might have changed since init due to curriculum training)
        ranges = None
        if isinstance(self.range, dict):
            ranges = list(self.range.values())
        else:
            ranges = [self.range]

        # Resample the command
        for i in range(self._command.shape[1]):
            self._command[env_ids, i] = num.uniform_(*ranges[i])
