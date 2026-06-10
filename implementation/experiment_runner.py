"""
Contains class definition of ExperimentRunner.
"""

from .mapf_instance import MAPFInstanceManager
from .mapf_utils import (EXPERIMENT_RESULTS_DIR,
                         GLOBAL_SOLVER_SEED,
                         MAX_NUM_INSTANCES)
from .wall_map import WallMap
from .pibt import PIBT
import pickle
from typing import Any
from copy import deepcopy
import numpy as np

class ExperimentRunner:
    """
    Represents a single experiment runner.

    One experiment runner equals data for one set of plots.
    """
    def __init__(self, *,
                 max_timestep: int,
                 hazard_config: str,
                 wall_map: WallMap,
                 even_or_random: str) -> None:
        self.socs: list[list[float]] = [[], []]
        self.makespans: list[list[float]] = [[], []]
        self.success_rates: list[list[float]] = [[], []]
        self.even_or_random = even_or_random
        self.manager = MAPFInstanceManager(max_timestep=max_timestep,
                                           hazard_config=hazard_config,
                                           wall_map=wall_map,
                                           even_or_random=even_or_random)
        self.solver = PIBT(dim=self.manager.wall_map.width,
                           seed=GLOBAL_SOLVER_SEED)

    def reset(self):
        """
        Reset manager, solver and data buffers.
        """
        for i in range(2):
            self.socs[i].clear()
            self.makespans[i].clear()
            self.success_rates[i].clear()
        self.manager.full_reset_all()
        self.solver.reset()

    def record_success_rate(self) -> None:
        """
        Records the success rate for hazard-aware and hazard-unaware planning
        over all instances and agent counts.
        """
        for _ in range(self.manager.get_max_num_agents()):
            tmp_successes: list[list[int]] = [[], []]

            for instance in self.manager.instances:
                self.solver.set_instance(instance)
                instance.add_agent()
                for hzd_type in range(2):
                    if hzd_type == 0:
                        self.solver.set_hazard_awareness(False)
                    else:
                        self.solver.set_hazard_awareness(True)

                    instance.reset()
                    self.solver.reset()

                    while not instance.finished():
                        instance.hazard_step()
                        self.solver.step()

                    tmp_successes[hzd_type].append(int(instance.succeeded()))

            # now we calculate success rate for a given num_agent
            for i in range(2):
                self.success_rates[i].append(sum(tmp_successes[i]) / MAX_NUM_INSTANCES)

    def record_soc_and_makespan(self) -> None:
        """
        Records normalized SoC and makespan for hazard-aware and hazard-unaware
        planning on a single instance across different agent counts.
        """
        # first, we record socs and makespan
        # we add one agent, test HA & HUA, record, then add another agent, ...
        # this we do only for the first instance

        instance = deepcopy(self.manager.instances[0])
        self.solver.set_instance(instance)
        for _ in range(self.manager.get_max_num_agents()):
            instance.add_agent()

            for hzd_type in range(2):
                if hzd_type == 0:
                    self.solver.set_hazard_awareness(False)
                else:
                    self.solver.set_hazard_awareness(True)

                instance.reset()
                self.solver.reset()

                while not instance.finished():
                    instance.hazard_step()
                    self.solver.step()

                if instance.succeeded():
                    self.socs[hzd_type].append(instance.path_manager.calc_soc() /
                                               instance.calc_optimal_soc(instance.num_agents))
                    self.makespans[hzd_type].append(
                        instance.path_manager.calc_makespan() /
                        instance.calc_optimal_makespan(instance.num_agents)
                    )
                    continue

                # if we failed (we only use successful episodes for soc & makespan)
                self.socs[hzd_type].append(np.nan)
                self.makespans[hzd_type].append(np.nan)

    def save(self) -> None:
        """
        Saves the recorded experiment data to disk.
        """
        data: dict[str, Any] = \
        {"socs": np.array(self.socs),
         "makespans": np.array(self.makespans),
         "success_rates": np.array(self.success_rates),
         "agents": np.array(list(range(1, self.manager.get_max_num_agents()+1)))}
        filepath = (
            EXPERIMENT_RESULTS_DIR /
            f"{self.manager.map_name}_{self.even_or_random}_{self.manager.hazard_name}"
        )
        pickle.dump(data, open(filepath, "wb"))

    def record_and_save_data(self) -> None:
        """
        Runs all experiments, records the results, and saves them to disk.
        """
        self.record_soc_and_makespan()
        #self.record_success_rate()
        self.save()

if __name__ == "__main__":
    runner = ExperimentRunner(max_timestep=300,
                              hazard_config="easy",
                              wall_map=WallMap("empty-8-8"),
                              even_or_random="random")

    runner.record_soc_and_makespan()
    runner.record_success_rate()
    runner.save()
