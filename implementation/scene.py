"""
Contains class definition of Scenes.
"""

from pathlib import Path
from .wall_map import WallMap
from .dist_table import DistTable
from .mapf_utils import (Position,
                         Positions,
                         MAX_NUM_SCENES,
                         get_scenario_path)

class SceneManager:
    """
    Manages multiple scenes.
    """
    def __init__(self, *,
                 wall_map: WallMap,
                 n_scenes: int,
                 even_or_random: str) -> None:
        assert 1 <= n_scenes <= MAX_NUM_SCENES

        self.scenes: list[Scene] = []
        for i in range(1, n_scenes+1):
            path = get_scenario_path(f"{wall_map.name}-{even_or_random}-{i}")
            self.scenes.append(Scene(wall_map=wall_map, path=path))

    @property
    def max_num_agents(self) -> int:
        """
        Return the maximum number of agents.

        Returns:
            int: Maximum number of agents
        """
        return self.scenes[0].max_num_agents

class Scene:
    """
    Represents a single scene file from mapf.info.
    """
    def __init__(self, *,
                 wall_map: WallMap,
                 path: Path) -> None:
        self.all_start_positions: Positions = []
        self.all_goal_positions: Positions = []
        self.all_dist_tables: list[DistTable] = []
        self.all_optimal_path_lenghts: list[float] = []
        self.max_num_agents: int = 0

        # read scenario file
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("version"):
                    continue

                parts = line.split()

                start = Position(int(parts[5]), int(parts[4]))
                goal = Position(int(parts[7]), int(parts[6]))

                self.all_start_positions.append(start)
                self.all_goal_positions.append(goal)
                table = DistTable(wall_map, goal)
                self.all_dist_tables.append(table)
                self.all_optimal_path_lenghts.append(table.get(start))
        self.max_num_agents = len(self.all_start_positions)

    def calc_optimal_makespan(self, for_n_agents: int) -> float:
        """
        Calculate the optimal makespan for the first n agents.

        The makespan is defined as the maximum optimal path length among the
        specified agents.

        Args:
            for_n_agents (int): Number of agents to consider.

        Returns:
            float: Optimal makespan.
        """
        return max(self.all_optimal_path_lenghts[:for_n_agents])

    def calc_optimal_soc(self, for_n_agents: int) -> float:
        """
        Calculate the optimal sum of costs for the first n agents.

        Args:
            for_n_agents (int): Number of agents to consider.

        Returns:
            float: Sum of optimal path lengths.
        """
        return sum(self.all_optimal_path_lenghts[:for_n_agents])
