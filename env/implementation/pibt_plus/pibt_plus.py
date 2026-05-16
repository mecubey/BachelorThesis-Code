"""Implementation of the PIBT+ algorithm."""

from ..grid import Grid
from ..pibt.pibt import PIBT
from ..header import FloatArr, Action, Configs, Actions, DIR_TO_ACT
from ..lacam.lacam import LaCAM
import numpy as np

class PIBTPLUS:
    """
    Implementation of the PIBT+ algorithm utilizing
    LaCAM as its' complement solver.
    """
    def __init__(self, grid: Grid, pibt_seed: int = 0, lacam_seed: int = 0) -> None:
        self.grid = grid
        self.pibt = PIBT(grid, pibt_seed)

        all_agent_distances: FloatArr = np.zeros(grid.num_agents, float)
        for i in grid.agent_idx:
            all_agent_distances[i] = self.pibt.dist_tables[i].get(grid.agent_positions[i])

        self.t_min = int(np.max(all_agent_distances))*2

        self.lacam = LaCAM(grid=grid, dist_tables=self.pibt.dist_tables, seed=lacam_seed)

    def step(self, t: int) -> list[Action]:
        """
        Uses PIBT to iteratively step through the enviroment.
        Once we have crossed T_min, we switch over to the
        complement solver (LaCAM). This solver receives a much easier
        problem then PIBT started out with.

        Args:
            t (int): Current timestep of the enviroment.

        Returns:
            list[Action]: Action of each agent.
        """
        if t < self.t_min:
            return self.pibt.step()

        if t == self.t_min:
            self.lacam.solve()
            print("LET'S GO!")

        return self.get_actions_from_configs(self.lacam.solutions, t)

    def get_actions_from_configs(self, solution: Configs, t: int) -> Actions:
        """
        Extract actions from given solutions.

        Args:
            solution (Configs): Set of solutions (paths).
            t (int): Current enviroment timestep.

        Returns:
            Actions: Next action for each agent.
        """
        current_t = t-self.t_min
        action_dict: Actions = []
        for i in self.grid.agent_idx:
            action_dict.append(DIR_TO_ACT[solution[current_t+1][i] - solution[current_t][i]])

        return action_dict
