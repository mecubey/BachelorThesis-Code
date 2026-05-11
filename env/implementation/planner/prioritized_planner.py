"""
Priority based planner using SIPP.
"""

from typing import cast
from .sipp import plan
from .. import header as h
from ..node import NodePath
from ..grid import Grid
from ..reservation_table import ReservationTable
import numpy as np

class PrioritizedPlanner():
    """
    Prioritized planner utilizing SIPP for multi agent path planning.
    """
    def __init__(self,
                 grid: Grid,
                 time_limit: int,
                 seed: int|None = None) -> None:
        self.grid: Grid = grid
        self.time_limit: int = time_limit
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.reservations: ReservationTable = ReservationTable(time_limit)
        self.paths: dict[str, NodePath] = {}

    def get_actions_at(self, t: int) -> dict[str, h.Action]:
        """Get the action of each agent at a specific timestep.

        Args:
            t (int): specified timestep

        Returns:
            dict[str, h.Action]: Action dictionary, has form {"agent_0": act_of_0, ...}
        """
        action_dictionary: dict[str, h.Action] = {}
        for a_name in self.grid.agent_names:
            action_dictionary[a_name] = self.paths[a_name].get_action_at(t)
        return action_dictionary

    def initial_plan(self) -> None:
        """
        Sets the plans for all agents at the start of the episode.\n
        Use get_actions_at to get actions for each agent.
        """
        entire_episode_interval = h.Interval(0, self.time_limit)

        # reserve starting positions for entire episode
        for pos in self.grid.agent_positions:
            self.reservations.reserve_interval(pos, entire_episode_interval)

        # reserve goal positions for entire episode
        for pos in self.grid.goal_positions:
            self.reservations.reserve_interval(pos, entire_episode_interval)

        # for now, random priorities
        for i in h.randomly(list(range(self.grid.num_agents)), self.rng):
            i = cast(int, i)

            # remove start position reservation of agent i
            self.reservations.clear_reserved_interval(self.grid.agent_positions[i],
                                                      entire_episode_interval)

            # remove goal position reservation of agent i
            self.reservations.clear_reserved_interval(self.grid.goal_positions[i],
                                                      entire_episode_interval)

            nodes = plan(grid=self.grid,
                         t=0,
                         time_limit=self.time_limit,
                         reservations=self.reservations,
                         start=self.grid.agent_positions[i],
                         goal=self.grid.goal_positions[i])

            self.paths[self.grid.agent_names[i]] = \
            NodePath.init_and_build_safe_intervals(nodes, self.reservations)
