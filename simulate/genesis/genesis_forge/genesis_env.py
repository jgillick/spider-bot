"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import math
import torch
import genesis as gs
from gymnasium import spaces
from genesis.engine.entities import RigidEntity
from typing import Any, Callable, Literal

EnvMode = Literal["train", "eval", "play"]


class GenesisEnv:
    """
    Base vectorized environment for Genesis.

    Args:
        num_envs: Number of parallel environments.
        dt: Simulation time step.
        max_episode_length_sec: Maximum episode length in seconds.
        max_episode_random_scaling: Scale the maximum episode length by this amount (+/-) so that not all environments reset at the same time.
        headless: Whether to run the environment in headless mode.
    """

    scene: gs.Scene = None
    robot: RigidEntity = None
    terrain: RigidEntity = None
    headless: bool = True
    num_envs: int = 1
    step_count: int = 0

    actions: torch.Tensor = None
    last_actions: torch.Tensor = None
    episode_length: torch.Tensor = None
    data_tracker_fn: Callable[[str, float], None] = None
    action_space: spaces.Space | None = None
    observation_space: spaces.Space | None = None

    max_episode_length: torch.Tensor = None
    """The max episode length, in steps, for each environment."""

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_sec: int | None = 10,
        max_episode_random_scaling: float = 0.0,
        headless: bool = True,
        mode: EnvMode = "train",
    ):
        self.dt = dt
        self.device = gs.device
        self.num_envs = num_envs
        self.headless = headless
        self.mode = mode

        self._max_episode_length_sec = 0.0
        self._max_episode_random_scaling = 0.0
        self._base_max_episode_length = None
        if max_episode_length_sec and max_episode_length_sec > 0:
            self._max_episode_random_scaling = max_episode_random_scaling / self.dt
            self.max_episode_length = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_int
            )
            self.max_episode_length[:] = self.set_max_episode_length(
                max_episode_length_sec
            )

    """
    Properties
    """

    @property
    def unwrapped(self):
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped :class:`GenesisEnv` instance
        """
        return self

    @property
    def max_episode_length_sec(self) -> int | None:
        """The max episode length, in seconds, for each environment."""
        return self._max_episode_length_sec

    """
    Utilities
    """

    def set_data_tracker(self, track_data_fn: Callable[[str, float], None]):
        """Define the data logger function."""
        self.data_tracker_fn = track_data_fn

    def track_data(self, name: str, value: float):
        """Log a single value to tensorboard, or similar"""
        if not self.data_tracker_fn:
            print(f"Warning: No logger function set for logging data.")
            return
        self.data_tracker_fn(name, value)

    def set_max_episode_length(self, max_episode_length_sec: int) -> int:
        """
        Set or change the maximum episode length.

        Args:
            max_episode_length_sec: The maximum episode length in seconds.

        Returns:
            The maximum episode length in steps.
        """
        self._max_episode_length_sec = max_episode_length_sec
        self._base_max_episode_length = math.ceil(max_episode_length_sec / self.dt)
        return self._base_max_episode_length

    """
    Operations
    """

    def construct_scene(
        self, rigid_options: gs.options.RigidOptions = None
    ) -> gs.Scene:
        """
        Construct the genesis scene.
        """
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=rigid_options
            or gs.options.RigidOptions(
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

    def build(self) -> None:
        """Builds the scene once all entities have been added (via construct_scene). This operation is required before running the simulation."""
        if self.scene is None:
            self.construct_scene()

        self.scene.build(n_envs=self.num_envs)

    def observations(self) -> torch.Tensor:
        """Generate a list of observations for each environment."""
        return torch.zeros((self.num_envs, 1), device=gs.device)

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
        self.step_count += 1
        self.episode_length += 1

        if self.actions is not None:
            self.last_actions[:] = self.actions[:]
        else:
            self.last_actions = torch.zeros_like(actions, device=gs.device)
        self.actions = actions

        return None, None, None, None, {}

    def reset(
        self,
        envs_idx: list[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or all parallel environments.

        Args:
            envs_idx: The environment ids to reset. If None, all environments are reset.

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # Initial reset, set buffers
        if self.step_count == 0:
            self.actions = torch.zeros(
                (self.num_envs, self.action_space.shape[0]),
                device=gs.device,
                dtype=gs.tc_float,
            )
            self.last_actions = torch.zeros_like(self.actions, device=gs.device)
            self.episode_length = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=torch.int32
            )

        # Actions
        if envs_idx.numel() > 0:
            self.actions[envs_idx] = 0.0
            self.last_actions[envs_idx] = 0.0

            # Episode length
            self.episode_length[envs_idx] = 0

        # Randomize max episode length for env_ids
        if (
            len(envs_idx) > 0
            and self._max_episode_random_scaling > 0.0
            and self._base_max_episode_length
        ):
            scale = torch.rand((envs_idx.numel(),)) * self._max_episode_random_scaling
            self.max_episode_length[envs_idx] = torch.round(
                self._base_max_episode_length + scale
            ).to(gs.tc_int)

        return None, None

    def close(self):
        """Close the environment."""
        self.scene.reset()

    def render(self):
        """Not implemented."""
        pass
