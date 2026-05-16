"""Priority Inheritance with Backtracking (PIBT) algorithm for MAPF."""

from .dist_table import DistTable
import numpy as np
from ..grid import Grid
from ..header import Position, DIR_TO_ACT, Action

class PIBT:
    """
    Priority Inheritance with Backtracking algorithm for MAPF.

    PIBT is an iterative algorithm that computes collision-free paths for
    multiple agents quickly, even with hundreds of agents or more. It uses
    priority inheritance and backtracking to resolve conflicts efficiently.

    The algorithm is sub-optimal but provides acceptable solutions almost
    immediately. It maintains distance tables for each agent to their goal
    and uses these for informed decision making. Priorities are dynamically
    updated based on progress toward goals.

    Completeness Guarantee:
        All agents are guaranteed to reach their destinations within a finite
        time when all pairs of adjacent vertices belong to a simple cycle 
        (i.e., biconnected). This property holds regardless of the number 
        of agents.

    Attributes:
        grid: Grid object representing the map.
        dist_tables: Distance tables for each agent to their goal.
        NIL: Sentinel value representing unassigned agent.
        NIL_COORD: Sentinel value representing unassigned coordinate.
        occupied_now: Current occupation status of each grid cell.
        occupied_nxt: Next timestep occupation status of each grid cell.
        rng: Random number generator for tie-breaking.

    References:
        Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
        Priority inheritance with backtracking for iterative multi-agent
        path finding. Artificial Intelligence Journal.
        https://kei18.github.io/pibt2/

    Note:
        PIBT serves as a core component in LaCAM (AAAI-23), which uses
        PIBT to quickly obtain initial solutions for eventually optimal
        multi-agent pathfinding. See https://kei18.github.io/lacam-project/
    """

    def __init__(self,
                 grid: Grid,
                 seed: int = 0) -> None:
        """Initialize PIBT solver.

        Args:
            grid: Grid object which represents map.
            seed: Random seed for tie-breaking (default: 0).
        """
        self.grid = grid

        # distance tables
        self.dist_tables = [DistTable(grid, goal) for goal in grid.goal_positions]

        # cache
        self.nil = grid.num_agents  # meaning \bot
        self.nil_coord: Position = Position(grid.dim, grid.dim)  # meaning \bot
        self.occupied_now = np.full((grid.dim, grid.dim), self.nil, dtype=int)
        self.occupied_nxt = np.full((grid.dim, grid.dim), self.nil, dtype=int)

        # define priorities
        self.priorities: list[float] = []
        for i in range(grid.num_agents):
            # agent with highest distance to goal get's highest priority
            self.priorities.append(self.dist_tables[i].get(grid.agent_positions[i]) / self.grid.dim)

        # define buffer for each agent action
        self.all_agent_actions: list[Action] = [Action.DO_NOTHING for _ in grid.agent_idx]

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def func_pibt(self, q_from: list[Position], q_to: list[Position], i: int) -> bool:
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
        c = sorted(c, key= lambda x: self.dist_tables[i].get(x))

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
            if (
                j != self.nil
                and (q_to[j] == self.nil_coord)
                and (not self.func_pibt(q_from, q_to, j))
            ):
                continue

            return True

        # failed to secure node
        q_to[i] = q_from[i]
        self.occupied_nxt[*q_from[i]] = i
        return False

    def step(self) -> list[Action]:
        """Compute next actions for all agents.

        Executes one timestep of PIBT by calling func_pibt for all agents
        in priority order.

        Returns:
            Next actions for each agent.
        """
        # setup
        q_to: list[Position] = []
        for i, v in enumerate(self.grid.agent_positions):
            q_to.append(self.nil_coord)
            self.occupied_now[*v] = i

        # perform PIBT
        a = sorted(list(range(self.grid.num_agents)),
                   key=lambda i: self.priorities[i], reverse=True)
        for i in a:
            if q_to[i] == self.nil_coord:
                self.func_pibt(self.grid.agent_positions, q_to, i)

        # cleanup & goal checking
        k: int = 0
        for from_pos, to_pos in zip(self.grid.agent_positions, q_to):
            self.occupied_now[*from_pos] = self.nil
            self.occupied_nxt[*to_pos] = self.nil
            if q_to[k] != self.grid.goal_positions[k]:
                self.priorities[k] += 1
            else:
                self.priorities[k] -= np.floor(self.priorities[k])
            self.all_agent_actions[k] = DIR_TO_ACT[to_pos-from_pos]
            k += 1

        return self.all_agent_actions
