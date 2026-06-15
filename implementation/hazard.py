"""
Contains class definitions of Hazard and HazardConfig.
"""

from dataclasses import dataclass
import yaml
import numpy as np
from .wall_map import WallMap
from .mapf_utils import (Position,
                         Map,
                         HazardType,
                         HAZARD_CONFIGS,
                         MAX_DAMAGE)

@dataclass
class HazardConfig:
    """
    Represents a hazard config.
    """
    name: str
    spawn_prob: float
    base_stuck_prob: float
    dir_spread_probs: list[float]
    add_damage_increase: float
    add_damage_decrease: float
    mult_damage_increase: float
    mult_damage_decrease: float
    growth_time: int
    stable_time: int

    @staticmethod
    def from_config(name: str) -> HazardConfig:
        """
        Read hazard config.

        Args:
            name (str): Name of hazard config.

        Returns:
            HazardConfig: Contents of the hazard config as a HazardConfig object.
        """
        with HAZARD_CONFIGS.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if name not in data:
            raise ValueError(f"Config '{name}' not found!")

        cfg = data[name]

        return HazardConfig(
            name=name,
            spawn_prob=cfg["spawnProb"],
            base_stuck_prob=cfg["baseStuckProb"],
            dir_spread_probs=cfg["dirSpreadProbs"],
            add_damage_increase=cfg["add_damage_increase"],
            add_damage_decrease=cfg["add_damage_decrease"],
            mult_damage_increase=cfg["mult_damage_increase"],
            mult_damage_decrease=cfg["mult_damage_decrease"],
            growth_time=cfg["growthTime"],
            stable_time=cfg["stableTime"]
        )

class Hazard:
    """
    Represents a dynamically spreading hazard.
    """
    def __init__(self, wall_map: WallMap, config: HazardConfig, seed: int) -> None:
        self.wall_map = wall_map
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.occupied_tiles: set[Position] = set()
        self.fronts: list[set[Position]] = []
        self.hazard_map = Map(wall_map.width, wall_map.height, "bool")
        self.spread_progress = 0
        self.stable_progress = 0
        self.hzd_type = HazardType.ADDITIVE

    def change_hzd_type(self, new_type: HazardType):
        """
        Change hazard type (ADDITIVE or MULTIPLICATIVE.)
        """
        self.hzd_type = new_type

    def step(self):
        """
        Step through hazard.
        """
        if len(self.occupied_tiles) == 0:
            self.spawn()
            return

        done_spreading = self.spread_progress >= self.config.growth_time

        if not done_spreading:
            self.spread()
            self.spread_progress += 1
            return

        # we finished spreading, so now we stabilize
        done_stable = self.stable_progress >= self.config.stable_time

        if done_stable:
            last_front = self.fronts.pop()
            for pos in last_front:
                self.hazard_map[pos] = False

            self.occupied_tiles.difference_update(last_front)

            if len(self.occupied_tiles) == 0:
                self.reset()
                return

        self.stable_progress += 1

    def spread(self) -> None:
        """
        Spread the hazard from currently occupied tiles to neighbouring tiles.

        Returns:
            Positions: All the directions this hazard spread towards.
        """
        assert len(self.occupied_tiles) > 0

        newly_occupied_tiles: set[Position] = set()
        for pos in self.occupied_tiles:
            neighbours = self.wall_map.neighbour_table[pos]
            for i, neighbour in enumerate(neighbours):
                if (neighbour is None or
                    self.on_hazard(neighbour)):
                    continue

                if self.rng.random() > self.config.dir_spread_probs[i]:
                    continue

                newly_occupied_tiles.add(neighbour)
                self.hazard_map[neighbour] = True
        self.occupied_tiles.update(newly_occupied_tiles)
        self.fronts.append(newly_occupied_tiles)

    def spawn(self) -> None:
        """
        Attempt to spawn a new hazard on a random free tile.

        Returns:
            bool: True if a new hazard was spawned, False otherwise.
        """
        assert len(self.occupied_tiles) == 0

        if self.rng.random() > self.config.spawn_prob:
            return
        idx = self.rng.integers(len(self.wall_map.free_tiles))
        spawn_pos: Position = self.wall_map.free_tiles[idx]
        self.occupied_tiles.add(spawn_pos)
        self.fronts.append(set([spawn_pos]))
        self.hazard_map[spawn_pos] = True

    def is_stuck(self, pos: Position) -> bool:
        """
        Determine whether a given position becomes stuck
        through the hazard.
        """
        if self.on_hazard(pos) and self.rng.random() < self.config.base_stuck_prob:
            return True
        return False

    def reset(self) -> None:
        """
        Reset the hazard simulation.
        """
        self.hazard_map.reset()
        self.occupied_tiles.clear()
        self.fronts.clear()
        self.spread_progress = 0
        self.stable_progress = 0

    def reseed(self, seed: int) -> None:
        """
        Reseed hazard RNG.
        """
        self.rng = np.random.default_rng(seed)

    def calculate_decreased_dmg(self, prev_dmg: float) -> float:
        """
        Calculate the next decreased hazard damage based on the current taken damage.
        """
        # for now, only additive
        match(self.hzd_type):
            case HazardType.ADDITIVE:
                return max(prev_dmg-self.config.add_damage_decrease, 0)
            case HazardType.MULTIPLICATIVE:
                return prev_dmg*(1-self.config.mult_damage_decrease)

    def calculate_increased_dmg(self, prev_dmg: float) -> float:
        """
        Calculate the next increased hazard damage based on the current taken damage.
        """
        # for now, only additive
        match(self.hzd_type):
            case HazardType.ADDITIVE:
                return min(prev_dmg+self.config.add_damage_increase, MAX_DAMAGE)
            case HazardType.MULTIPLICATIVE:
                return min(prev_dmg*(1+self.config.mult_damage_increase), MAX_DAMAGE)

    def on_hazard(self, pos: Position):
        """
        Check whether a given position is currently marked as a hazard tile.
        """
        return self.hazard_map[pos]

if __name__ == "__main__":
    from pprint import pprint

    test_config = HazardConfig.from_config("additive_easy")

    test_map = WallMap("empty-8-8")

    hazard = Hazard(test_map, test_config, 0)

    hazard.step()

    while True:
        hazard.step()
        pprint(hazard.hazard_map.tiles.reshape(8, 8).astype(int))
        input()
