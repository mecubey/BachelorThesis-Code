"""
Contains implementation of a lazy BFS distance table.

Used in PIBT.
"""

from collections import deque
from dataclasses import dataclass, field
from ..grid import Grid
from ..header import Position

import numpy as np

@dataclass
class DistTable:
    """Distance table for computing shortest path distances using BFS.

    This class lazily evaluates distances from a goal position to any target
    position on the grid using breadth-first search (BFS). Distances are
    cached for efficient repeated queries.

    Attributes:
        grid: 2D boolean array where True indicates free space.
        goal: Goal position (y, x) coordinates.
        Q: Queue for BFS traversal (lazy distance evaluation).
        table: Distance matrix storing computed distances.
    """
    grid: Grid
    goal: Position
    q: deque[Position] = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self) -> None:
        """Initialize distance table with goal position."""
        self.q = deque([self.goal])
        self.table = np.full((self.grid.dim, self.grid.dim), np.inf, dtype=float)
        self.table[*self.goal] = 0

    def get(self, target: Position) -> float:
        """Get shortest path distance from goal to target.

        Uses lazy BFS evaluation to compute distance on demand. Previously
        computed distances are cached in the table.

        Args:
            target: Target position (x, y) coordinates.

        Returns:
            Shortest path distance from goal to target. Returns np.inf
            if target is invalid or unreachable.
        """
        # check valid input
        if self.grid.contains_wall(target):
            return np.inf

        # distance has been known
        if self.table[*target] < np.inf:
            return self.table[*target]

        # BFS with lazy evaluation
        while len(self.q) > 0:
            u = self.q.popleft()
            d = self.table[*u]
            for v in self.grid.get_neighbours_at(u):
                if d + 1 < self.table[*v]:
                    self.table[*v] = d + 1
                    self.q.append(v)
            if u == target:
                return d

        return np.inf
