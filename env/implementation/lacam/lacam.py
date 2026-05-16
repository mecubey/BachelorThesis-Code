"""Implementation of the LaCAM algorithm."""

from collections import deque
import numpy as np
from .node import LowLevelNode, HighLevelNode
from ..pibt.dist_table import DistTable
from .pibt import PIBT
from ..grid import Grid
from ..header import Config, Configs

class LaCAM:
    """Implementation of the LaCAM algorithm."""
    def __init__(self, *,
                 grid: Grid,
                 dist_tables: list[DistTable],
                 seed: int = 0) -> None:
        self.grid: Grid = grid
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

        # set distance tables (taken from PIBT)
        self.dist_tables = dist_tables

        self.pibt = PIBT(self.dist_tables, grid, seed)

        self.solutions: Configs = []

    def solve(self) -> None:
        """
        Solve a given MAPF instance with LaCAM

        Sets the lists of positions each agent should occupy for the next
        timesteps.
        """
        # set search scheme
        open_hln: deque[HighLevelNode] = deque([])
        explored_hln: dict[Config, HighLevelNode] = {}
        n_goal: HighLevelNode | None = None

        # set initial node
        q_init = self.grid.agent_positions
        n_init = HighLevelNode(
            q=q_init, order=self.get_order(q_init), h=self.get_h_value(q_init)
        )
        open_hln.appendleft(n_init)
        explored_hln[n_init.q] = n_init

        # main loop
        while len(open_hln) > 0:
            n: HighLevelNode = open_hln[0]

            # goal check
            if n_goal is None and n.q == self.grid.goal_positions:
                n_goal = n
                self.solutions = self.backtrack(n_goal)
                return

            # lower bound check
            if n_goal is not None and n_goal.g <= n.f:
                open_hln.popleft()
                continue

            # low-level search end
            if len(n.tree) == 0:
                open_hln.popleft()
                continue

            # low-level search
            c: LowLevelNode = n.tree.popleft()  # constraints
            if c.depth < self.grid.num_agents:
                i = n.order[c.depth]
                v = n.q[i]
                cands = [v] + self.grid.get_neighbours_at(v)
                self.rng.shuffle(cands)
                for u in cands:
                    n.tree.append(c.get_child(i, u))

            # generate the next configuration
            q_to = self.configuration_generator(n, c)
            if q_to is None:
                # invalid configuration
                continue

            if q_to in explored_hln:
                # known configuration
                n_known = explored_hln[q_to]
                n.neighbors.add(n_known)
                open_hln.appendleft(n_known)  # typically helpful
            else:
                # new configuration
                n_new = HighLevelNode(
                    q=q_to,
                    parent=n,
                    order=self.get_order(q_to),
                    g=n.g + self.get_edge_cost(n.q, q_to),
                    h=self.get_h_value(q_to),
                )
                n.neighbors.add(n_new)
                open_hln.appendleft(n_new)
                explored_hln[q_to] = n_new

    def backtrack(self, _n: HighLevelNode | None) -> Configs:
        """
        Backtrack to get paths for each agent.
        """
        configs: Configs = []
        n = _n
        while n is not None:
            configs.append(n.q)
            n = n.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, q_from: Config, q_to: Config) -> int:
        """
        E.g., \\sum_i | not (Q_from[i] == Q_to[k] == g_i).
        """
        cost = 0
        for i in range(self.grid.num_agents):
            if not self.grid.goal_positions[i] == q_from[i] == q_to[i]:
                cost += 1
        return cost

    def get_h_value(self, q: Config) -> float:
        """
        E.g., \\sum_i dist(Q[i], g_i).
        """
        cost = 0
        for agent_idx, loc in enumerate(q):
            cost += self.dist_tables[agent_idx].get(loc)
        return cost

    def get_order(self, q: Config) -> list[int]:
        """
        Get order by descending order of dist(Q[i], g_i).

        Note that this is not an effective PIBT prioritization scheme
        Args:
            q (Config): Distances to goals.

        Returns:
            list[int]: Order in which agents are priotized (highest priority agent is first).
        """
        order = list(range(self.grid.num_agents))
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(q[i]), reverse=True)
        return order

    def configuration_generator(self,
                                n: HighLevelNode,
                                c: LowLevelNode) -> Config | None:
        """
        Generate next configuration according to constraint tree.
        """
        # setup next configuration
        q_to = Config([self.pibt.nil_coord for _ in range(self.grid.num_agents)])
        for k in range(c.depth):
            q_to[c.who[k]] = c.where[k]

        # apply PIBT
        success = self.pibt.step(n.q, q_to, n.order)
        return q_to if success else None
