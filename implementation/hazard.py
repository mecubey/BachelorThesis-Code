"""
Contains class definitions of Hazard and HazardConfig.
"""

from .wall_map import WallMap
from .mapf_utils import (Positions,
                        Position,
                        Map,
                        HAZARD_CONFIGS)
from dataclasses import dataclass
import yaml
import numpy as np

@dataclass
class HazardConfig:
    """
    Represents a hazard config.
    """
    name: str
    lifetime: int
    spawn_prob: float
    base_stuck_prob: float
    dir_spread_probs: list[float]

    @staticmethod
    def from_config(name: str) -> HazardConfig:
        """
        Read hazard config.

        Args:
            name (str): Name of hazard config.

        Returns:
            HazardConfig: Contents of the hazard config as HazardConfig object.
        """
        with HAZARD_CONFIGS.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if name not in data:
            raise ValueError(f"Config '{name}' not found!")

        cfg = data[name]

        return HazardConfig(
            name=name,
            lifetime=cfg["lifetime"],
            spawn_prob=cfg["spawnProb"],
            base_stuck_prob=cfg["baseStuckProb"],
            dir_spread_probs=cfg["dirSpreadProbs"],
        )

class Hazard:
    """
    Represents a dynamically spreading hazard.
    """
    def __init__(self, wall_map: WallMap, config: HazardConfig, seed: int) -> None:
        self.wall_map = wall_map
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.occupied_tiles: Positions = []
        self.hazard_map = Map(wall_map.width, wall_map.height, "bool")
        self.spread_progress = 0

    def spread(self):
        """
        Spread the hazard from currently occupied tiles to neighbouring tiles.

        For each occupied tile, this method checks all neighbouring tiles and
        probabilistically expands the hazard based on the directional spread
        probabilities. Newly infected tiles are collected and added after the
        iteration completes. If any spread occurs, the progression counter is
        advanced.
        """
        newly_occupied_tiles: Positions = []
        for pos in self.occupied_tiles:
            neighbours = self.wall_map.neighbour_table[pos]
            for i, neighbour in enumerate(neighbours):
                if (neighbour is None or
                    self.on_hazard(neighbour) or
                    self.rng.random() > self.config.dir_spread_probs[i]):
                    continue
                newly_occupied_tiles.append(neighbour)
                self.hazard_map[neighbour] = True
        self.occupied_tiles.extend(newly_occupied_tiles)

        self.progress()

    def spawn(self) -> bool:
        """
        Attempt to spawn a new hazard on a random free tile.

        A tile is selected uniformly at random from all free tiles. The spawn
        succeeds based on the configured spawn probability. If successful, the
        tile becomes occupied and the simulation progress is advanced.

        Returns:
            bool: True if a new hazard was spawned, False otherwise.
        """
        if self.rng.random() > self.config.spawn_prob:
            return False
        idx = self.rng.integers(len(self.wall_map.free_tiles))
        spawn_pos: Position = self.wall_map.free_tiles[idx]
        self.occupied_tiles.append(spawn_pos)
        self.hazard_map[spawn_pos] = True
        self.progress()
        return True

    def empty(self) -> bool:
        """
        Check whether there are currently no active hazard tiles.

        Returns:
            bool: True if no tiles are occupied by hazards, False otherwise.
        """
        return len(self.occupied_tiles) == 0

    def done(self) -> bool:
        """
        Check whether the hazard simulation has reached its maximum spread limit.

        Returns:
            bool: True if the spread progression has reached the configured limit,
            False otherwise.
        """
        return self.spread_progress >= self.config.lifetime

    def is_stuck(self, pos: Position) -> bool:
        """
        Determine whether a given position becomes stuck
        through the hazard.

        Args:
            pos (Position): Position to evaluate.

        Returns:
            bool: True if the position is stuck, False otherwise.
        """
        if self.rng.random() < self.calc_stuck_prob(pos):
            return True
        return False

    def reset(self) -> None:
        """
        Reset the hazard simulation state.

        Clears all occupied tiles, resets the hazard map, and resets the
        spread progression counter.
        """
        self.hazard_map.reset()
        self.occupied_tiles.clear()
        self.spread_progress = 0

    def progress(self) -> None:
        """
        Advance the internal spread progression counter by one step.
        """
        self.spread_progress += 1

    def on_hazard(self, pos: Position):
        """
        Check whether a given position is currently marked as a hazard tile.

        Args:
            pos (Position): Position to check.

        Returns:
            bool: True if the position is currently a hazard, False otherwise.
        """
        return self.hazard_map[pos]

    def calc_stuck_prob(self, pos: Position) -> float:
        """
        Calculate te probability of getting stuck on this tile.

        Args:
            pos (Position): Position to be checkded.

        Returns:
            _type_: Probabibility of getting stuck. 0 if pos is not a hazard tile.
        """
        if not self.on_hazard(pos):
            return 0
        return self.config.base_stuck_prob

if __name__ == "__main__":
    import numpy as np

    test_config = HazardConfig.from_config("test")

    test_map = WallMap("empty-8-8")

    hazard = Hazard(test_map, test_config, 0)

    hazard.spawn()

    while True:
        for row in range(hazard.hazard_map.width):
            for col in range(hazard.hazard_map.height):
                print(f"{hazard.hazard_map[Position(row, col)]} ", end="")
            print()
        input()
        hazard.spread()
