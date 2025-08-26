import re
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class ContactManager(BaseManager):
    """
    Tracks the contact forces between entity links in the environment.

    Args:
        env: The environment to track the contact forces for.
        link_names: The names, or name regex patterns, of the entity links to track the contact forces for.
        entity: The entity which contains the links we're tracking. Defaults to `env.robot`.
        track_air_time: Whether to track the air time of the entity link contacts.
        air_time_contact_threshold: When track_air_time is True, this is the threshold for the contact forces to be considered.
        with_entity: Filter the contact forces to only include contacts with this entity.
        with_links_names: Filter the contact forces to only include contacts with these links.

    Example:
        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def configuration_managers(self):
                self.contact_manager = ContactManager(
                    self,
                    link_names=[".*_Foot"],
                )

            def step(self, actions: torch.Tensor):
                super().step(actions)

                self.contact_manager.step()

                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: list[int] | None = None):
                super().reset(envs_idx)

                self.contact_manager.reset(envs_idx)

                return obs, info

            def calculate_rewards():
                # Reward for each foot in contact
                CONTACT_THRESHOLD = 1.0
                CONTACT_WEIGHT = 0.005
                has_contact = self.contact_manager.contacts[:,:].norm(dim=-1) > CONTACT_THRESHOLD
                contact_reward = has_contact.sum(dim=1).float() * CONTACT_WEIGHT

                # ...additional reward calculations here...
    """

    contacts: torch.Tensor | None = None
    """Contact forces experienced by the entity links."""

    last_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air before the last contact."""

    current_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air since the last detach."""

    last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach."""

    current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact."""

    def __init__(
        self,
        env: GenesisEnv,
        link_names: list[str],
        entity: RigidEntity = None,
        track_air_time: bool = False,
        air_time_contact_threshold: float = 1.0,
        with_entity: RigidEntity = None,
        with_links_names: list[int] = None,
    ):
        super().__init__(env)
        self._link_names = link_names
        self._air_time_contact_threshold = air_time_contact_threshold
        self._track_air_time = track_air_time

        # Get the link indices
        entity = entity or env.robot
        self._target_link_ids = self._get_links_idx(entity, link_names)
        self._with_link_ids = None
        if with_entity or with_links_names:
            with_entity = with_entity or env.robot
            self._with_link_ids = self._get_links_idx(with_entity, with_links_names)

        # Initialize buffers
        link_count = len(self._target_link_ids)
        self.contacts = torch.zeros((env.num_envs, link_count, 3), device=gs.device)
        if self._track_air_time:
            self.last_air_time = torch.zeros(
                (self.env.num_envs, link_count), device=gs.device
            )
            self.current_air_time = torch.zeros_like(self.last_air_time)
            self.last_contact_time = torch.zeros_like(self.last_air_time)
            self.current_contact_time = torch.zeros_like(self.last_air_time)

    """
    Helper Methods
    """

    def get_first_contact(self, dt: float, time_margin: float = 1.0e-8) -> torch.Tensor:
        """Checks if links that have established contact within the last :attr:`dt` seconds.

        This function checks if the links have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the links are considered to be in contact.

        Args:
            dt: The time period since the contact was established.
            time_margin: The absolute time margin for the time period

        Returns:
            A boolean tensor indicating the links that have established contact within the last
            :attr:`dt` seconds. Shape is (n_envs, n_target_links)

        Raises:
            RuntimeError: If the manager is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the manager configuration."
            )
        # check if the bodies are in contact
        currently_in_contact = self.current_contact_time > 0.0
        less_than_dt_in_contact = self.current_contact_time < (dt + time_margin)
        return currently_in_contact * less_than_dt_in_contact

    def get_first_air(self, dt: float, time_margin: float = 1.0e-8) -> torch.Tensor:
        """Checks links that have broken contact within the last :attr:`dt` seconds.

        This function checks if the links have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the links are considered to not be in contact.

        Args:
            dt: The time period since the contact was broken.
            time_margin: The absolute time margin for the time period

        Returns:
            A boolean tensor indicating the links that have broken contact within the last
            :attr:`dt` seconds. Shape is (n_envs, n_target_links)

        Raises:
            RuntimeError: If the manager is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self._track_air_time:
            raise RuntimeError(
                "The contact manager is not configured to track contact time."
                "Please enable the 'track_air_time' in the manager configuration."
            )
        currently_detached = self.current_air_time > 0.0
        less_than_dt_detached = self.current_air_time < (dt + time_margin)
        return currently_detached * less_than_dt_detached

    """
    Operations
    """

    def step(self):
        super().step()
        self._calculate_contact_forces()
        self._calculate_air_time()

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # reset the current air time
        if self._track_air_time:
            self.current_air_time[envs_idx] = 0.0
            self.current_contact_time[envs_idx] = 0.0
            self.last_air_time[envs_idx] = 0.0
            self.last_contact_time[envs_idx] = 0.0

    """
    Implementation
    """

    def _get_links_idx(
        self, entity: RigidEntity, names: list[str] = None
    ) -> torch.Tensor:
        """
        Find the global link indices for the given link names or regular expressions.

        Args:
            entity: The entity to find the links in.
            names: The names, or name regex patterns, of the links to find.

        Returns:
            List of global link indices.
        """
        # If link names are not defined, assume all links
        if names is None:
            return [link.idx for link in entity.links]

        ids = []
        for pattern in names:
            try:
                # Find link by name
                link = entity.get_link(pattern)
                if link is not None:
                    ids.append(link.idx)
            except:
                # Find link by regex
                for link in entity.links:
                    if re.match(pattern, link.name):
                        ids.append(link.idx)

        return torch.tensor(ids, device=gs.device)

    def _calculate_contact_forces(self):
        """
        Reduce all contacts down to a single force vector for each target link.

        Args:
            collision_data: Dict with 'force', 'link_a', 'link_b' tensors
            target_link_ids: List of link IDs to accumulate forces for
            with_link_ids: Optional list of link IDs to filter contacts by.
                        Only contacts involving these links will be considered.

        Returns:
            Tensor of shape (n_envs, n_target_links, 3)
        """
        contacts = self.env.scene.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True
        )
        force = contacts["force"]
        link_a = contacts["link_a"]
        link_b = contacts["link_b"]

        # Filter contacts by with_link_ids if specified
        if self._with_link_ids is not None:
            with_links = self._with_link_ids.to(force.device)
            n_with_links = with_links.shape[0]
            with_links = with_links.view(1, 1, n_with_links)

            # Check if either link_a or link_b is in with_link_ids
            link_a_expanded = link_a.unsqueeze(-1)
            link_b_expanded = link_b.unsqueeze(-1)

            mask_with_a = (link_a_expanded == with_links).any(dim=-1)
            mask_with_b = (link_b_expanded == with_links).any(dim=-1)
            contact_filter = mask_with_a | mask_with_b

            # Apply filter to all tensors
            force = force * contact_filter.unsqueeze(-1)

        # Concatenate links and forces - each force applies to both links in the pair
        all_links = torch.cat([link_a, link_b], dim=1)
        all_forces = torch.cat([force, force], dim=1)

        # Convert target_link_ids to tensor for broadcasting
        target_links = self._target_link_ids.to(force.device)
        n_target_links = target_links.shape[0]
        target_links = target_links.view(1, 1, n_target_links)

        # Create mask for where each target link appears
        all_links = all_links.unsqueeze(-1)
        mask = all_links == target_links

        # Apply mask and sum
        force_expanded = all_forces.unsqueeze(-2)
        masked_forces = force_expanded * mask.unsqueeze(-1)
        self.contacts = masked_forces.sum(dim=1)

    def _calculate_air_time(self):
        """
        Track air time values for the links
        """
        if not self._track_air_time:
            return

        dt = self.env.scene.dt

        # Check contact state of bodies
        is_contact = (
            torch.norm(self.contacts[:, :, :], dim=-1)
            > self._air_time_contact_threshold
        )
        is_first_contact = (self.current_air_time > 0) * is_contact
        is_first_detached = (self.current_contact_time > 0) * ~is_contact

        # Update the last contact time if body has just become in contact
        self.last_air_time = torch.where(
            is_first_contact,
            self.current_air_time + dt,
            self.last_air_time,
        )

        # Increment time for bodies that are not in contact
        self.current_air_time = torch.where(
            ~is_contact,
            self.current_air_time + dt,
            0.0,
        )

        # Update the last contact time if body has just detached
        self.last_contact_time = torch.where(
            is_first_detached,
            self.current_contact_time + dt,
            self.last_contact_time,
        )

        # Increment time for bodies that are in contact
        self.current_contact_time = torch.where(
            is_contact,
            self.current_contact_time + dt,
            0.0,
        )
