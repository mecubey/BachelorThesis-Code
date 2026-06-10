"""
This file actually runs the experiments
and saves their data for later plotting.
"""

import sys
sys.path.insert(0, '')

from implementation.experiment_runner import ExperimentRunner
from implementation.wall_map import WallMap

MAX_TIMESTEP = 500

wall_maps = [WallMap("random-32-32-10")]

config_names = ["tar", "syrup", "mud", "oil"]

runners: list[ExperimentRunner] = []

for wall_map in wall_maps:
    new_runner = ExperimentRunner(max_timestep=MAX_TIMESTEP,
                                  hazard_config="tar", # dummy value
                                  wall_map=wall_map,
                                  even_or_random="random")
    runners.append(new_runner)

for config in config_names:
    for runner in runners:
        runner.manager.change_hazard_configs(config)
        runner.manager.full_reset_all()
        runner.record_and_save_data()
