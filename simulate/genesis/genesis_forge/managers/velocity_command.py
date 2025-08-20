from typing import Tuple, Sequence, TypedDict

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import robot_lin_vel
from genesis_forge.managers.base import BaseManager

Range = Tuple[float, float]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


class VelocityCommandManager(BaseManager):
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
                    lin_vel_x_range=(-0.5, 0.5),
                    lin_vel_y_range=(-0.5, 0.5),
                    ang_vel_z_range=(-1.0, 1.0),
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
        lin_vel_x_range: The range of linear velocity in the x-direction
        lin_vel_y_range: The range of linear velocity in the y-direction
        ang_vel_z_range: The range of angular velocity in the z-direction
        resample_time_s: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer
    """

    def __init__(
        self,
        env: GenesisEnv,
        lin_vel_x_range: Range,
        lin_vel_y_range: Range,
        ang_vel_z_range: Range,
        resample_time_s: float = 5.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: DebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env)
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}

        self._arrow_nodes: list = []
        self._command = torch.zeros(env.num_envs, 3, device=gs.device)
        self._resample_steps = int(resample_time_s / env.dt)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self._command

    def step(self):
        """The environment has been stepped"""

        # Resample commands, if necessary
        resample_command_envs = (
            (self.env.episode_length % self._resample_steps == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_command(resample_command_envs)

        # Render arrows
        self._render_arrows()

    def reset(self, env_ids: Sequence[int] = None):
        """One or more environments have been reset"""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        """Create a new velocity commands for the given environment ids."""
        num = torch.empty(len(env_ids), device=gs.device)
        self._command[env_ids, 0] = num.uniform_(*self.lin_vel_x_range)
        self._command[env_ids, 1] = num.uniform_(*self.lin_vel_y_range)
        self._command[env_ids, 2] = num.uniform_(*self.ang_vel_z_range)

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
            *self.lin_vel_x_range, *self.lin_vel_y_range
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
        vec = self._command.clone()
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
