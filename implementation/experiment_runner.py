"""
Contains class definition of ExperimentRunner.
"""

import pickle
from typing import Any
from statistics import mean
import numpy as np
from .mapf_instance import MAPFInstance
from .mapf_utils import (EXPERIMENT_DIR,
                         GLOBAL_SOLVER_SEED,
                         GLOBAL_HAZARD_SEED,
                         MAX_NUM_SCENES)
from .scene import SceneManager
from .hazard import HazardConfig
from .wall_map import WallMap
from .pibt import PIBT

class ExperimentRunner:
    """
    Represents a single experiment runner.

    One experiment runner equals data for one set of plots.
    """
    def __init__(self, *,
                 max_timestep: int,
                 hazard_config: str,
                 wall_map: WallMap,
                 max_agents: int,
                 agent_size_step: int,
                 even_or_random: str) -> None:
        self.optimal_socs: list[list[float]] = [[], []]
        self.raw_socs: list[list[float]] = [[], []]
        self.optimal_makespans: list[list[float]] = [[], []]
        self.raw_makespans: list[list[float]] = [[], []]
        self.success_rates: list[list[float]] = [[], []]

        self.max_agents = max_agents
        self.agent_size_step = agent_size_step
        self.even_or_random = even_or_random
        self.scene_manager = SceneManager(wall_map=wall_map,
                                          n_scenes=MAX_NUM_SCENES,
                                          even_or_random=even_or_random)
        self.instance = MAPFInstance(max_timestep=max_timestep,
                                     hazard_config=HazardConfig.from_config(hazard_config),
                                     hazard_seed=GLOBAL_HAZARD_SEED,
                                     wall_map=wall_map,
                                     scene=self.scene_manager.scenes[0])
        self.solver = PIBT(width=wall_map.width,
                           height=wall_map.height,
                           seed=GLOBAL_SOLVER_SEED)
        self.solver.set_instance(self.instance)

    def reset(self):
        """
        Reset instance, solver and data buffers.
        """
        for i in range(2):
            self.optimal_socs[i].clear()
            self.raw_socs[i].clear()
            self.optimal_makespans[i].clear()
            self.raw_makespans[i].clear()
            self.success_rates[i].clear()
        self.instance.full_reset()
        self.solver.reset()

    def change_hazard_config(self, config: str) -> None:
        """
        Change current hazard config of this runner.

        Args:
            config (str): New config.
        """
        self.instance.change_hazard_config(HazardConfig.from_config(config))

    def record_data(self) -> None:
        """
        Records data for one set of plots.
        """
        for agent_count in range(self.agent_size_step, self.max_agents+1, self.agent_size_step):
            tmp_optimal_socs: list[list[float]] = [[], []]
            tmp_raw_socs: list[list[float]] = [[], []]
            tmp_optimal_makespans: list[list[float]] = [[], []]
            tmp_raw_makespans: list[list[float]] = [[], []]
            tmp_success_rates: list[list[float]] = [[], []]
            for i in range(MAX_NUM_SCENES):
                self.instance.full_reset()
                self.instance.change_scene(self.scene_manager.scenes[i])
                for _ in range(agent_count):
                    self.instance.add_agent()

                for hzd_type in [False, True]:
                    self.solver.set_hazard_awareness(hzd_type)

                    self.instance.reset()
                    self.solver.reset()

                    while not self.instance.finished():
                        self.instance.hazard_step()
                        self.instance.move_all_agents(self.solver.step())
                        self.instance.progress()

                    if self.instance.succeeded():
                        soc = self.instance.path_manager.calc_soc()
                        makespan = self.instance.path_manager.calc_makespan()
                        tmp_raw_socs[hzd_type].append(soc)
                        tmp_raw_makespans[hzd_type].append(makespan)

                        rel_optimal_soc = soc / self.instance.scene.calc_optimal_soc(agent_count)
                        rel_optimal_makespan = (makespan /
                                                self.instance.scene
                                                .calc_optimal_makespan(agent_count))
                        tmp_optimal_socs[hzd_type].append(rel_optimal_soc)
                        tmp_optimal_makespans[hzd_type].append(rel_optimal_makespan)

                    tmp_success_rates[hzd_type].append(self.instance.succeeded())

            for j in range(2):
                if not tmp_optimal_socs[j]:
                    self.optimal_socs[j].append(np.nan)
                    self.raw_socs[j].append(np.nan)
                else:
                    self.optimal_socs[j].append(mean(tmp_optimal_socs[j]))
                    self.raw_socs[j].append(mean(tmp_raw_socs[j]))

                if not tmp_optimal_makespans[j]:
                    self.optimal_makespans[j].append(np.nan)
                    self.raw_makespans[j].append(np.nan)
                else:
                    self.optimal_makespans[j].append(mean(tmp_optimal_makespans[j]))
                    self.raw_makespans[j].append(mean(tmp_raw_makespans[j]))

                self.success_rates[j].append(mean(tmp_success_rates[j]))

    def save(self) -> None:
        """
        Saves the recorded experiment data to disk.
        """
        agent_x_axis = list(range(self.agent_size_step, self.max_agents+1, self.agent_size_step))
        data: dict[str, Any] = \
        {"socs": np.array(self.optimal_socs),
         "raw_socs": np.array(self.raw_socs),
         "makespans": np.array(self.optimal_makespans),
         "raw_makespans": np.array(self.raw_makespans),
         "success_rates": np.array(self.success_rates),
         "agents": np.array(agent_x_axis)}
        filepath = (
            EXPERIMENT_DIR / "results" /
            (f"{self.instance.wall_map.name}" +
             f"_{self.even_or_random}_({self.instance.hazard.config.name})" +
             f"_{self.instance.hazard.hzd_type}")
        )
        pickle.dump(data, open(filepath, "wb"))
