"""
Logger utility class to collect data for experiments.
"""

from collections import defaultdict

class Logger:
    """
    Logger utility class to collect experiment data.
    """
    def __init__(self) -> None:
        self.hazard_dmg_buffer: dict[int, list[float]] = defaultdict(list)
        self.cost_of_paths_buffer: dict[int, list[float]] = defaultdict(list)
        self.episode_finish: int = 0
        self.makespan: int = 0
        self.shortest_path_lengths_buffer: dict[int, float] = defaultdict(int)

    def record_episode_end(self, *, fin: int) -> None:
        """
        Record whether or not the episode has finished.

        Args:
            episode_i (int): Current episode.
            fin (int): Flag for episode finish.
        """
        self.episode_finish = fin

    def record_makespan(self, *, makespan: int) -> None:
        """
        Record makespawn. -1 if episode did not finish.

        Args:
            episode_i (int): Current episode.
            makespan (int): Makespawn of episode.
        """
        self.makespan = makespan

    def record_shortest_path_cost(self, *,
                                  agent_i: int,
                                  distance: float) -> None:
        """
        Record the shortest path cost of the specified agent.

        Args:
            episode_i (int): Current episode.
            agent_i (int): Current agent,
            distance (int): Length of shortest path from initial agent position to agent's goal.
        """
        self.shortest_path_lengths_buffer[agent_i] = distance

    def record_last_move_cost(self, *,
                              agent_i: int,
                              last_move_cost: float):
        """
        Record cost of an agent's last move.

        Args:
            episode_i (int): Current episode.
            agent_i (int): Current agent.
            last_move_cost (float): Cost of the last move taken.
        """
        self.cost_of_paths_buffer[agent_i].append(last_move_cost)

    def record_hzd_dmg(self, *,
                       agent_i: int,
                       hazard_dmg: float) -> None:
        """
        Record an agent's taken hazard damage.

        Args:
            episode_i (int): Current episode.
            agent_i (int): Current agent.
            hazard_dmg (float): Hazard damage taken in this step.
        """
        self.hazard_dmg_buffer[agent_i].append(hazard_dmg)
