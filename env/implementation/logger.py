"""
Logger utility class to collect data for experiments.
"""

from .header import Statistic
import numpy as np

class Logger:
    """
    Logger utility class to collect experiment data.
    """
    def __init__(self) -> None:
        self.total_hzd_dmg: float = 0
        self.soc: float = 0
        self.did_finish: float = 0
        self.makespan: float = 0

    def get_statistics(self) -> Statistic:
        """
        Get total hazard dmg, SOC, makespan and
        finish status from this episode.
        """
        return Statistic(hazard_dmg=self.total_hzd_dmg,
                         soc=self.soc,
                         makespan=np.nan if not self.did_finish else self.makespan,
                         fin=self.did_finish)

    def reset(self) -> None:
        """
        Reset logger values.
        """
        self.total_hzd_dmg = 0
        self.soc = 0
        self.did_finish = 0
        self.makespan = 0

    def record_episode_end(self, fin: float) -> None:
        """
        Record whether or not the episode has finished.
        """
        self.did_finish = fin

    def record_makespan(self, makespan: float) -> None:
        """
        Record makespawn. -1 if episode did not finish.
        """
        self.makespan = makespan

    def record_last_move_cost(self, last_move_cost: float):
        """
        Add the cost of the last move to the SOC.
        """
        self.soc += last_move_cost

    def record_hzd_dmg(self, hazard_dmg: float) -> None:
        """
        Add hazard damage to the total hazard damage.
        """
        self.total_hzd_dmg += hazard_dmg
