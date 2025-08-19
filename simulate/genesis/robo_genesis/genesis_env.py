"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import math
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity
from gymnasium import spaces
from typing import Sequence, Any, Callable


class GenesisEnv:
    """
    Base vectorized environment for Genesis.
    """

    scene: gs.Scene = None
    robot: RigidEntity = None
    terrain: RigidEntity = None
    action_space: spaces = None
    observation_space: spaces = None
    headless: bool = True
    num_envs: int = 1
    num_steps: int = 0

    actions: torch.Tensor = None
    last_actions: torch.Tensor = None
    episode_length: torch.Tensor = None

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int = 15,
        headless: bool = True,
    ):
        self.dt = dt
        self.device = gs.device
        self.num_envs = num_envs
        self.headless = headless
        self.max_episode_length = math.ceil(max_episode_length_s / self.dt)

    def construct_scene(self) -> gs.Scene:
        """
        Construct the genesis scene.
        """
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

        # Add plane
        self.terrain = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        return self.scene

    def build_scene(self) -> None:
        """Builds the scene once all entities have been added (via construct_scene). This operation is required before running the simulation."""
        self.scene.build(n_envs=self.num_envs, env_spacing=(2, 2))

    def get_observations(self) -> torch.Tensor:
        """
        Get the observations for all parallel environments.
        """
        raise NotImplementedError

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Take an action for each parallel environment.

        Args:
            actions: Batch of actions with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        self.num_steps += 1
        self.episode_length += 1

        if self.actions is not None:
            self.last_actions[:] = self.actions[:]
        else:
            self.last_actions = torch.zeros_like(actions)
        self.actions = actions

        return None, None, None, None, {}

    def reset(
        self,
        env_ids: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or all parallel environments.

        Args:
            env_ids: The environment ids to reset. If None, all environments are reset.

        Returns:
            A batch of observations and info from the vectorized environment.
        """

        # Initial reset, set buffers
        if self.num_steps == 0:
            self.actions = torch.zeros(
                (self.num_envs, self.action_space.shape[0]),
                device=gs.device,
                dtype=gs.tc_float,
            )
            self.last_actions = torch.zeros_like(self.actions)
            self.episode_length = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_int
            )

        # Actions
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        # Episode length
        self.episode_length[env_ids] = 0

        return None, None

    def close(self):
        """Close the environment."""
        self.scene.reset()

    def render(self):
        """Not implemented."""
        pass

    def set_data_tracker(self, _track_data_fn: Callable[[str, float], None]):
        """Set the function which logs data to tensorboard."""
        pass
