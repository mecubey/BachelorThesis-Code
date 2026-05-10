"""
Contains zone related classes, attributes, methods.
"""

import numpy as np
from .grid import Grid
from . import header as h

class Zone():
    """
    Holds zone attributes.\n 
    The zone spreads in each timestep according to
    a specified probability and damages agents upon touching them.
    """
    def __init__(self, *,
                 grid: Grid,
                 free_tiles: h.IntArr,
                 dir_spread_probs: list[float],
                 max_num_spread: int,
                 seed: int|None = None) -> None:
        self.grid = grid
        self.free_tiles = free_tiles
        self.num_free_tiles = len(free_tiles)
        self.dir_spread_probs = dir_spread_probs
        self.max_num_spread = max_num_spread
        self.rng = np.random.default_rng(seed)
        self.spread_progress = 0
        self.occupied_tiles: list[h.Position] = []

    def spawn(self) -> None:
        """
        Spreads to a free position anywhere in the grid.

        occupied_tiles should be empty before calling this method.
        """
        assert self.empty(), \
            f"occupied_tiles needs to be empty before spawning, got {self.occupied_tiles}"
        start_pos = h.Position(*self.free_tiles[self.rng.choice(self.num_free_tiles)])
        self.grid.set_zone_in_grid(start_pos, True)
        self.occupied_tiles.append(start_pos)
        self.progress()

    def spread(self) -> list[h.Position]:
        """
        Randomly spread a zone tile. A zone tile can only spread to its
        neighbouring tiles.

        Returns positions of new tiles that were spread to.
        """
        newly_occupied_tiles: list[h.Position] = []
        for pos in self.occupied_tiles:
            for act in  h.ACTS_ARR[:-1]: # last action is "DO_NOTHING"
                new_pos: h.Position = pos+h.ACT_TO_DIR[act]

                # cannot spread to tiles that already contain zone,
                # cannot spread if prob does not hit,
                # cannot spread if on wall tile
                if self.grid.contains_zone(new_pos) or \
                   self.rng.random() > self.dir_spread_probs[act] or \
                   self.grid.contains_wall(new_pos):
                    continue

                newly_occupied_tiles.append(new_pos)
                self.grid.set_zone_in_grid(new_pos, True)

        self.occupied_tiles.extend(newly_occupied_tiles)

        return newly_occupied_tiles

    def empty(self) -> bool:
        """
        Returns true if the zone has not spread once yet.\n
        False otherwise.
        """
        return self.spread_progress == 0

    def progress(self) -> None:
        """
        Progress spread of hazard.
        """
        self.spread_progress += 1

    def done(self) -> bool:
        """
        Returns true if zone can't spread any more in the current timestep.\n
        False otherwise.
        """
        return self.spread_progress == self.max_num_spread

    def reset(self) -> None:
        """
        Remove all currently occupied tiles.\n
        Note that this does not modify the actual grid.
        """
        for pos in self.occupied_tiles:
            self.grid.set_zone_in_grid(pos, False)
        self.occupied_tiles = []
        self.spread_progress = 0
