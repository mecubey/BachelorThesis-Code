"""
Contains class definition of PathManager.
"""

from .mapf_utils import (Position,
                         Positions,
                         STAY)

class PathManager:
    """
    Manager class for agent paths.
    """
    def __init__(self, max_num_agents: int) -> None:
        self.paths: list[Positions] = [[] for _ in range(max_num_agents)]

    def reset(self) -> None:
        """
        Clears all stored paths.

        Removes all actions from each path in self.paths, effectively resetting
        the PathManager to an empty state.
        """
        for path in self.paths:
            path.clear()

    def append_action_to_path(self, i: int, action: Position):
        """
        Appends the next action to the corresponding path.

        Args:
            i (int): Index of agent.
            action (Positions): Next action of agent i.
        """
        self.paths[i].append(action)

    def calc_soc(self) -> int:
        """
        Computes the sum of costs (SoC) across all paths.

        The SoC is defined as the total number of actions across all paths,
        excluding trailing STAY actions at the end of each path.

        Returns:
            int: The sum of costs across all paths after removing trailing STAY actions.
        """
        soc: int = 0

        for path in self.paths:
            path_len = len(path)
            if path_len == 0:
                continue

            soc += path_len
            for action in reversed(path):
                if action == STAY:
                    soc -= 1
                    continue
                break
        return soc

    def calc_makespan(self) -> int:
        """
        Computes the makespan of all paths.

        Assumes all paths have equal length, as agents continue acting until
        the episode terminates.

        Returns:
            int: The length of any path, representing the makespan.
        """
        return len(self.paths[0])
