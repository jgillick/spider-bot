import torch
from typing import Any
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import RewardManager, TerminationManager, ContactManager
from genesis_forge.managers.action import BaseActionManager
from genesis_forge.managers.command import CommandManager


class ManagedEnvironment(GenesisEnv):
    """
    An environment where most of the logic is handled by managers.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.action_manager = PositionalActionManager(
                    self,
                    joint_names=".*",
                    pd_kp=50,
                    pd_kv=0.5,
                    max_force=8.0,
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                )
                self.reward_manager = RewardManager(
                    self,
                    term_cfg={
                        "Default pose": {
                            "weight": -1.0,
                            "fn": rewards.dof_similar_to_default,
                            "params": {
                                "dof_action_manager": self.action_manager,
                            },
                        },
                        "Base height": {
                            "fn": mdp.rewards.base_height,
                            "params": { "target_height": 0.135 },
                            "weight": -100.0,
                        },
                    },
                )

            def step(self, actions: torch.Tensor):
                _, rewards, terminated, truncated, info = super().step(actions)
                obs = self.observations()
                return obs, rewards, terminated, truncated, info

            def observations(self) -> torch.Tensor:
                return torch.cat(
                    [
                        ...
                    ]
                )


    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_manager: BaseActionManager | None = None
        self.reward_managers: list[RewardManager] = []
        self.termination_managers: list[TerminationManager] = []
        self.contact_managers: list[ContactManager] = []
        self.command_managers: list[CommandManager] = []

        self._reward_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self._terminated_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self._truncated_buf = torch.zeros_like(self._terminated_buf)

    """
    Properties
    """

    @property
    def action_space(self) -> torch.Tensor:
        """The action space, provided by the action manager if it exists."""
        if self.action_manager is not None:
            return self.action_manager.action_space
        return None

    """
    Managers
    """

    def add_action_manager(self, manager: BaseActionManager):
        """
        Adds an action manager to the environment.
        """
        assert (
            self.action_manager is None
        ), "An action manager already exists, and an environment cannot have multiple action managers."
        self.action_manager = manager

    def add_reward_manager(self, manager: RewardManager):
        """
        Adds a reward manager to the environment.
        """
        self.reward_managers.append(manager)

    def add_termination_manager(self, manager: TerminationManager):
        """
        Adds a termination manager to the environment.
        """
        self.termination_managers.append(manager)

    def add_contact_manager(self, manager: ContactManager):
        """
        Adds a contact manager to the environment.
        """
        self.contact_managers.append(manager)

    def add_command_manager(self, manager: CommandManager):
        """
        Adds a command manager to the environment.
        """
        self.command_managers.append(manager)

    """
    Operations
    """

    def build(self):
        """Called when the scene is built"""
        super().build()
        if self.action_manager is not None:
            self.action_manager.build()
        for contact_manager in self.contact_managers:
            contact_manager.build()
        for termination_manager in self.termination_managers:
            termination_manager.build()
        for reward_manager in self.reward_managers:
            reward_manager.build()
        for command_manager in self.command_managers:
            command_manager.build()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Performs a step in the environment.
        """
        super().step(actions)

        # Execute the actions and a simulation step
        if self.action_manager is not None:
            self.action_manager.step(actions)
        self.scene.step()

        # Calculate contact forces
        for contact_manager in self.contact_managers:
            contact_manager.step()

        # Calculate termination and truncation
        self._terminated_buf[:] = False
        self._truncated_buf[:] = False
        for termination_manager in self.termination_managers:
            terminated, truncated = termination_manager.step()
            self._terminated_buf[:] |= terminated
            self._truncated_buf[:] |= truncated
        dones = self._terminated_buf | self._truncated_buf
        reset_env_idx = dones.nonzero(as_tuple=False).reshape((-1,))

        # Calculate rewards
        self._reward_buf[:] = 0.0
        for reward_manager in self.reward_managers:
            self._reward_buf += reward_manager.step()

        # Command managers
        for command_manager in self.command_managers:
            command_manager.step()

        # Reset environments
        if reset_env_idx.numel() > 0:
            self.reset(reset_env_idx)

        return None, self._reward_buf, self._terminated_buf, self._truncated_buf, {}

    def reset(
        self, env_ids: list[int] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset managers."""
        (obs, info) = super().reset(env_ids)

        if self.action_manager is not None:
            self.action_manager.reset(env_ids)
        for contact_manager in self.contact_managers:
            contact_manager.reset(env_ids)
        for termination_manager in self.termination_managers:
            termination_manager.reset(env_ids)
        for reward_manager in self.reward_managers:
            reward_manager.reset(env_ids)
        for command_manager in self.command_managers:
            command_manager.reset(env_ids)

        return (obs, info)
