"""
Contains zone related classes, attributes, methods.
"""

from typing import Callable
import numpy as np
from . import header as h

class Zone():
    """
    Holds zone attributes.\n 
    The zone spreads in each timestep according to
    a specified probability and damages agents upon touching them.
    """
    def __init__(self, *,
                 dir_spread_probs: list[float],
                 max_num_spread: int) -> None:
        self.dir_spread_probs = dir_spread_probs
        self.spread_progress = 0
        self.max_num_spread = max_num_spread
        self.occupied_tiles: list[h.PositionT] = []

    def spawn(self, *, start_pos: h.PositionT) -> None:
        """
        Sets the first tile of this zone.\n
        occupied_tiles should be empty before calling this method.\n
        Note that this does not modify the actual grid.
        """
        assert self.empty(), \
            f"occupied_tiles needs to be empty before spawning, got {self.occupied_tiles}"
        self.occupied_tiles.append(start_pos)

    def spread(self, *,
               rng: np.random.Generator,
               on_zone: Callable[[h.PositionT], bool],
               no_wall: Callable[[h.PositionT], bool],
               inside_grid: Callable[[h.PositionT], bool]) -> list[h.PositionT]:
        """
        Randomly spread a zone tile. A zone tile can only spread to its
        neighbouring tiles. Note that this does not modify the actual grid.

        Returns positions of new tiles that were spread to.
        """
        self.spread_progress += 1

        newly_occupied_tiles: list[h.PositionT] = []
        for pos in self.occupied_tiles:
            for i in range(h.ACTION_LEN-1): # last action is "DO_NOTHING"
                new_pos: h.PositionT = pos+h.Act_To_Dir_Arr[i]

                # cannot spread to tiles that already contain zone,
                # cannot spread if prob does not hit,
                # cannot spread if on wall tile,
                # cannot spread if new_pos is outside grid
                if on_zone(new_pos) or \
                   rng.random() > self.dir_spread_probs[i] or \
                   not no_wall(new_pos) or \
                   not inside_grid(new_pos):
                    continue

                newly_occupied_tiles.append(new_pos)

        self.occupied_tiles.extend(newly_occupied_tiles)

        return newly_occupied_tiles

    def empty(self) -> bool:
        """
        Returns true if there are no tiles which the zone
        has spread to. False otherwise.
        """
        return not self.occupied_tiles

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
        self.occupied_tiles = []
        self.spread_progress = 0
