from typing import Tuple, Sequence, TypedDict

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import robot_lin_vel

from .command_manager import CommandManager, CommandRange


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class VelocityCommandRange(TypedDict):
    lin_vel_x: CommandRange
    lin_vel_y: CommandRange
    ang_vel_z: CommandRange


class DebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: Sequence[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    arrow_offset: float
    """The vertical offset of the debug arrows from the top of the robot"""

    arrow_radius: float
    """The radius of the shaft of the debug arrows"""

    arrow_max_length: float
    """The maximum length of the debug arrows"""

    commanded_color: Tuple[float, float, float, float]
    """The color of the commanded velocity arrow"""

    actual_color: Tuple[float, float, float, float]
    """The color of the actual robot velocity arrow"""


DEFAULT_VISUALIZER_CONFIG: DebugVisualizerConfig = {
    "envs_idx": None,
    "arrow_offset": 0.01,
    "arrow_radius": 0.02,
    "arrow_max_length": 0.15,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
    "actual_color": (0.0, 0.0, 0.5, 1.0),
}


class VelocityCommandManager(CommandManager):
    """
    Generates a velocity command from uniform distribution.
    The command comprises of a linear velocity in x and y direction and an angular velocity around the z-axis.

    To use the manager:
        1. Create the manager in your environment's init method
        2. Call it in your step method (`self.command_manager.step()`)
        3. Call it in your reset method  (`self.command_manager.reset(env_ids)`)
        4. Get the target velocity command with the `command` property for your reward and observation functions

    Example:
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.command_manager = VelocityCommandManager(
                    self,
                    visualize=True,
                    range = {
                        "lin_vel_x_range": (-1.0, 1.0),
                        "lin_vel_y_range": (-1.0, 1.0),
                        "ang_vel_z_range": (-0.5, 0.5),
                    }
                )

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
                target_cmd = self.command_manager.command

                # Tracking of linear velocity commands (xy axes)
                actual_vel = self.robot.get_vel()
                lin_vel_error = torch.sum(
                    torch.square(target_cmd[:, :2] - actual_vel[:, :2]), dim=1
                )
                line_vel_reward = torch.exp(-lin_vel_error / 0.25)

                # Tracking of angular velocity commands (yaw)
                ang_vel_error = torch.square(target_cmd[:, 2] - self.base_ang_vel[:, 2])
                ang_vel_reward = torch.exp(-ang_vel_error / 0.25)

                # ...additional reward calculations here...

    Debug Visualization:
        If you set `debug_visualizer` to True, arrows will be rendered above your robot
        showing the commanded velocity vs the actual velocity.
        The commanded velocity is green and the actual velocity is blue.

    Args:
        env: The environment to control
        range: The ranges of linear & angular velocities
        standing_probability: The probability of all velocities being zero for an environment
        resample_time_s: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: VelocityCommandRange,
        resample_time_s: float = 5.0,
        standing_probability: float = 0.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: DebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_s=resample_time_s)
        self._arrow_nodes: list = []
        self.standing_probability = standing_probability
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}

        self._is_standing_env = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

    def step(self):
        """Render the command arrows"""
        super().step()
        self._render_arrows()

    def _resample_command(self, env_ids: Sequence[int]):
        """Overwrites commands for environments that should be standing still."""
        super()._resample_command(env_ids)

        num = torch.empty(len(env_ids), device=gs.device)
        self._is_standing_env[env_ids] = (
            num.uniform_(0.0, 1.0) <= self.standing_probability
        )
        standing_envs_idx = self._is_standing_env.nonzero(as_tuple=False).flatten()
        self._command[standing_envs_idx, :] = 0.0

    def _render_arrows(self):
        """Render the command arrows"""
        if not self.debug_visualizer:
            return

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            self.env.scene.clear_debug_object(arrow)
        self._arrow_nodes = []

        # Scale the arrow size based on the maximum target velocity range
        scale_factor = self.visualizer_cfg["arrow_max_length"] / max(
            *self.range["lin_vel_x"], *self.range["lin_vel_y"], *self.range["ang_vel_z"]
        )

        # Calculate the center of the robot
        aabb = self.env.robot.get_AABB()
        max_aabb = aabb[:, 1]  # [max_x, max_y, max_z]
        min_aabb = aabb[:, 0]  # [min_x, min_y, min_z]
        robot_x = (min_aabb[:, 0] + max_aabb[:, 0]) / 2
        robot_y = (min_aabb[:, 1] + max_aabb[:, 1]) / 2
        robot_z = max_aabb[:, 2]

        # Set the arrow position over the center of the robot
        arrow_pos = torch.zeros(self.env.num_envs, 3, device=gs.device)
        arrow_pos[:, 0] = robot_x
        arrow_pos[:, 1] = robot_y
        arrow_pos[:, 2] = robot_z + self.visualizer_cfg["arrow_offset"]
        arrow_pos += torch.from_numpy(self.env.scene.envs_offset).to(gs.device)

        # Convert velocity command to vector direction
        vec = self.command.clone()
        vec[:, 2] = 0.0
        vec[:, :] *= scale_factor

        # Actual robot velocity
        actual_vec = robot_lin_vel(self.env).clone()
        actual_vec[:, 2] = 0.0
        actual_vec[:, :] *= scale_factor

        debug_envs = (
            self.visualizer_cfg["envs_idx"]
            if self.visualizer_cfg["envs_idx"] is not None
            else range(self.env.num_envs)
        )
        for i in debug_envs:
            # Target arrow
            self.draw_arrow(
                pos=arrow_pos[i],
                vec=vec[i],
                color=self.visualizer_cfg["commanded_color"],
            )
            # Actual arrow
            self.draw_arrow(
                pos=arrow_pos[i],
                vec=actual_vec[i],
                color=self.visualizer_cfg["actual_color"],
            )

    def draw_arrow(
        self,
        pos: torch.Tensor,
        vec: torch.Tensor,
        color: Sequence[float],
    ):
        # If velocity is zero, don't draw the arrow
        if torch.all(vec == 0.0):
            return
        try:
            node = self.env.scene.draw_debug_arrow(
                pos=pos.cpu().numpy(),
                vec=vec.cpu().numpy(),
                color=color,
                radius=self.visualizer_cfg["arrow_radius"],
            )
            if node:
                self._arrow_nodes.append(node)
            else:
                print("No node returned")
        except Exception as e:
            print(f"Error adding debug visualizing in VelocityCommandManager: {e}")
