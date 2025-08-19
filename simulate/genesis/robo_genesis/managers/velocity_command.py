import os
from re import I
import torch
import genesis as gs
from typing import Tuple, Sequence
from genesis.engine.entities import RigidEntity

from robo_genesis.genesis_env import GenesisEnv

Range = Tuple[float, float]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ARROW_VEC_MAX = 0.15


class VelocityCommandManager:
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


    """

    env: GenesisEnv
    lin_vel_x_range: Range
    lin_vel_y_range: Range
    ang_vel_z_range: Range
    arrow_offset: float = 0.01
    visualize: bool

    _command: torch.Tensor = None
    _arrow_nodes: list = ()
    _command_resample_steps: int

    def __init__(
        self,
        env: GenesisEnv,
        lin_vel_x_range: Range,
        lin_vel_y_range: Range,
        ang_vel_z_range: Range,
        arrow_offset: float = 0.01,
        resample_time_s: float = 5.0,
        visualize: bool = False,
    ):
        self.env = env
        self.visualize = visualize
        self.arrow_offset = arrow_offset
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range

        self._command = torch.zeros(env.num_envs, 3, device=gs.device)
        self.resample_steps = int(resample_time_s / env.dt)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self._command

    def construct_scene(self):
        """Add the arrows to the scene"""

        def make_arrow(color: Sequence[float]):
            return self.env.scene.add_entity(
                gs.morphs.Mesh(
                    file=os.path.join(THIS_DIR, "../assets/arrow.stl"),
                    pos=[0.0, 0.0, 0.0],
                    quat=[1.0, 0.0, 0.0, 0.0],
                    scale=self.arrow_scale,
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=color,
                    ),
                ),
            )

        self.target_arrow = make_arrow((0.0, 0.5, 0.0, 0.0))
        self.actual_arrow = make_arrow((0.0, 0.0, 0.5, 0.0))

    def step(self):
        """The environment has been stepped"""

        # Resample commands, if necessary
        resample_command_envs = (
            (self.env.episode_length % self.resample_steps == 0)
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
        if not self.visualize:
            return

        scene_context = self.env.scene.visualizer.context

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            scene_context.clear_debug_object(arrow)
        self._arrow_nodes = []

        # Scale the arrow size based on the maximum target velocity range
        scale_factor = ARROW_VEC_MAX / max(*self.lin_vel_x_range, *self.lin_vel_y_range)

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
        arrow_pos[:, 2] = robot_z + self.arrow_offset
        arrow_pos += self.env.scene.envs_offset

        # Convert velocity command to vector direction
        vec = self._command.clone()
        vec[:, 2] = 0.0
        vec[:, :] *= scale_factor

        # Actual robot velocity
        actual_vec = self.env.robot.get_vel().clone()
        actual_vec[:, 2] = 0.0
        actual_vec[:, :] *= scale_factor

        for i in range(self.env.num_envs):
            # Target arrow
            self._arrow_nodes.append(
                scene_context.draw_debug_arrow(
                    pos=arrow_pos[i],
                    vec=vec[i],
                    color=[0.0, 0.5, 0.0],
                    radius=0.025,
                )
            )
            # Actual arrow
            self._arrow_nodes.append(
                scene_context.draw_debug_arrow(
                    pos=arrow_pos[i],
                    vec=actual_vec[i],
                    color=[0.0, 0.0, 0.5],
                    radius=0.026,
                )
            )
