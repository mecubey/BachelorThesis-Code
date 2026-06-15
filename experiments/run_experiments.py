"""
This file actually runs the experiments
and saves their data for later plotting.
"""

import sys
sys.path.insert(0, '')
from implementation.experiment_runner import ExperimentRunner
from implementation.wall_map import WallMap
from implementation.mapf_utils import HazardType

MAX_TIMESTEP = 500

wall_maps = [WallMap("random-32-32-10")]

config_names = ["SlowFire",
                "Wildfire",
                "Minefield"]

runners: list[ExperimentRunner] = []


first_runner = ExperimentRunner(
                    max_timestep=MAX_TIMESTEP,
                    hazard_config="SlowFire",
                    wall_map=wall_maps[0],
                    max_agents=450,
                    agent_size_step=50,
                    even_or_random="random"
                )

"""second_runner = ExperimentRunner(
                    max_timestep=MAX_TIMESTEP,
                    hazard_config="SlowFire",
                    wall_map=wall_maps[1],
                    max_agents=1000,
                    agent_size_step=50,
                    even_or_random="random"
                )"""

runners = [first_runner]

for runner in runners:
    for config_name in config_names:
        runner.change_hazard_config(config_name)
        for hzd_type in [HazardType.ADDITIVE, HazardType.MULTIPLICATIVE]:
            runner.instance.hazard.change_hzd_type(hzd_type)
            runner.reset()
            runner.record_data()
            runner.save()
