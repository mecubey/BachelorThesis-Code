"""
Slightly modified PIBT to use as configuration generator in LaCAM.
"""

import numpy as np
from ..pibt.dist_table import DistTable
import numpy as np
from ..grid import Grid
from ..header import Position, Config

class PIBT:
    """
    Priority Inheritance with Backtracking algorithm for MAPF.

    Sligtly modified so it can be used in LaCAM to generate new configuraitons.
    """
    def __init__(self, dist_tables: list[DistTable], grid: Grid, seed: int = 0) -> None:
        self.grid = grid
        self.dist_tables = dist_tables

        # cache
        self.nil = grid.num_agents # meaning \bot
        self.nil_coord: Position = Position(grid.dim, grid.dim) # meaning \bot
        self.occupied_now = np.full((grid.dim, grid.dim), self.nil, dtype=int)
        self.occupied_nxt = np.full((grid.dim, grid.dim), self.nil, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def func_pibt(self,
                 q_from: Config,
                 q_to: Config,
                 i: int) -> bool:
        """
        Core PIBT function for single agent planning with priority inheritance.

        Attempts to assign a collision-free next position for agent i. If
        another agent j occupies the desired position, recursively invokes
        PIBT for agent j (priority inheritance). Backtracks if no valid
        position is found.

        Args:
            Q_from: Current configuration (positions at current timestep).
            Q_to: Next configuration being constructed (modified in-place).
            i: Agent index to plan for.

        Returns:
            True if successfully assigned a position to agent i, False otherwise.
        """
        # true -> valid, false -> invalid

        # get candidate next vertices
        c = [q_from[i]] + self.grid.get_neighbours_at(q_from[i])
        self.rng.shuffle(c)  # tie-breaking, randomize
        c = sorted(c, key=self.dist_tables[i].get)

        # vertex assignment
        for v in c:
            # avoid vertex collision
            if self.occupied_nxt[*v] != self.nil:
                continue

            j = self.occupied_now[*v]

            # avoid edge collision
            if j != self.nil and q_to[j] == q_from[i]:
                continue

            # reserve next location
            q_to[i] = v
            self.occupied_nxt[*v] = i

            # priority inheritance (j != i due to the second condition)
            if (j != self.nil
                and (q_to[j] == self.nil_coord)
                and (not self.func_pibt(q_from, q_to, j))):
                continue

            return True

        # failed to secure node
        q_to[i] = q_from[i]
        self.occupied_nxt[*q_from[i]] = i
        return False

    def step(self,
             q_from: Config,
             q_to: Config,
             order: list[int],) -> bool:
        """Compute next configuration."""
        flg_success = True

        # setup
        for i, (v_i_from, v_i_to) in enumerate(zip(q_from, q_to)):
            self.occupied_now[*v_i_from] = i
            if v_i_to != self.nil_coord:
                #  check vertex collision
                if self.occupied_nxt[*v_i_to] != self.nil:
                    flg_success = False
                    break
                # check edge collision
                j = self.occupied_now[*v_i_to]
                if j != self.nil and j != i and q_to[j] == v_i_from:
                    flg_success = False
                    break
                self.occupied_nxt[*v_i_to] = i

        # perform PIBT
        if flg_success:
            for i in order:
                if q_to[i] == self.nil_coord:
                    flg_success = self.func_pibt(q_from, q_to, i)
                    if not flg_success:
                        break

        # cleanup
        for from_pos, to_pos in zip(q_from, q_to):
            self.occupied_now[*from_pos] = self.nil
            if to_pos != self.nil_coord:
                self.occupied_nxt[*to_pos] = self.nil

        return flg_success
