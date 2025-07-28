"""Compute metrics (averaged over the episode) and log them at reset"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerBase, ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class MetricsTermCfg(ManagerTermBaseCfg):
    """Configuration for a metric term."""
    pass

class MetricManager(ManagerBase):
    """Manager for computing metric signals for a given world."""

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the metric manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, MetricsTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[MetricsTermCfg] = list()
        self._class_term_cfgs: list[MetricsTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual metric term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Buffer which stores the current step metric for each term for each environment
        self._step_metric = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for metri manager."""
        msg = f"<MetricManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Metric Terms"
        table.field_names = ["Index", "Name"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, _) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual metric terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual metric terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual metric terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            extras["Episode_Metric/" + key] = torch.mean(self._episode_sums[key][env_ids])
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the metric terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """
        Computes the metric signal for a step
        """
        # iterate over all the metric terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params)
            # update episodic sum
            self._episode_sums[name] += value

            # Update current metric for this step.
            self._step_metric[:, term_idx] = value

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: MetricsTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the metric term.
            cfg: The configuration for the metric term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Metric term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> MetricsTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the metric term.

        Returns:
            The configuration of the metric term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Metric term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_metric[env_idx, idx].cpu().item()]))
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, MetricsTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MetricsTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)