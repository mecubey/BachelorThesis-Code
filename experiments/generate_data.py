"""
Contains methods to generate data for plotting.
"""

import sys
sys.path.insert(0, '')

from copy import deepcopy
import os
import pickle
import numpy as np
from env.implementation.header import (HazardDamageType,
                                       EnvParams,
                                       Statistic)
from env.implementation.path_task_env import PathTaskMultiAgentEnv
from env.implementation.pibt.pibt import PIBT
from env.path_task_env_v0 import raw_env
from env.implementation.pregenerate import all_free_tiles, wall_maps
from statistics import mean
from experiment_utils import (Parameter,
                              PlannerType,
                              Metric,
                              Data,
                              NUM_SPLITS,
                              BASE_SPLITS,
                              AGENTS_SPLITS,
                              DIR_SPREAD_SPLITS,
                              SEEDS,
                              default_params)

data_path = os.path.dirname(os.path.abspath(__file__)) + "/data/"

def set_param_value(*,
                    varying_param: Parameter,
                    env_params: EnvParams,
                    val: int) -> None:
    """
    Set parameter value of the experiment enviroment parameters.
    """
    match(varying_param):
        case Parameter.SPAWN_PROB:
            env_params.spawn_prob = BASE_SPLITS[val]
        case Parameter.DIR_SPREAD_PROB:
            env_params.dir_spread_probs = DIR_SPREAD_SPLITS[val]
        case Parameter.SPREAD_PROB:
            env_params.spread_prob = BASE_SPLITS[val]
        case Parameter.AGENTS:
            env_params.num_agents = AGENTS_SPLITS[val]

def run_experiment(*,
                   map_idx: int,
                   seed: int,
                   env: PathTaskMultiAgentEnv) -> Statistic:
    """
    For a given seed and enviroment parameters,
    calculate the SOC, accumulated hazard damage,
    success rate, makespan.

    Args:
        params (EnvParams): Enviroment parameters.
        episode_seeds (list[int]): Set of seeds.

    Returns:
        Statistic:
        Total hazard damage,
        SoC,
        finish status,
        makespan.
    """
    env.reset(wall_map=wall_maps[map_idx],
              free_tiles=all_free_tiles[map_idx],
              env_seed=seed,
              zone_seed=seed)

    planner = PIBT(grid=env.grid,
                   zone=env.zone,
                   with_decay=env.args.with_decay,
                   consider_hazards=env.args.consider_hazards,
                   seed=seed)

    done = False
    while not done:
        actions_dict = planner.step()
        termination, truncation = env.step(actions_dict)
        done = termination or truncation

    return env.logger.get_statistics()

def gen_data(*,
             map_id: int,
             with_decay: bool,
             varying_param: Parameter|None = None,
             dmg_type: HazardDamageType = HazardDamageType.CONSTANT) -> str:
    """
    Run experiments and pickles the data.

    Args:
        map_id (int): Map id.
        num_splits (int): Into how many values the specified parameter should be split.
        seeds (list[int]): How many seeds are run for each split value.
        varying_param (Parameter | None, optional): Parameter to be varied. Defaults to None.
        dmg_type (HazardDamageType, optional): Specified damage type. Defaults to Constant.
    Returns:
        Path of experiment directory.
    """
    # we don't want to modify the default parameters
    env_params = deepcopy(default_params)

    env_params.hazard_dmg_type = dmg_type
    env_params.with_decay = with_decay

    env = raw_env(env_params)

    data: Data = {m: {"avg": {t: [] for t in PlannerType},
                      "min": {t: [] for t in PlannerType},
                      "max": {t: [] for t in PlannerType}} for m in Metric}

    for t in PlannerType:
        if t == PlannerType.HAZARD_UNAWARE:
            # now we collect data for not hazard aware planner
            env_params.consider_hazards = False

        for i in range(NUM_SPLITS):
            if varying_param is not None:
                set_param_value(varying_param=varying_param,
                                env_params=env_params,
                                val=i)

            seed_hazard_dmgs: list[float] = []
            seed_socs: list[float] = []
            seed_makespans: list[float] = []
            seed_fins: list[float] = []

            for s in SEEDS:
                statistic = run_experiment(env=env,
                                           map_idx=map_id,
                                           seed=s+(i*len(SEEDS)))

                # only use successfull episodes for quality measures
                if statistic.fin:
                    seed_hazard_dmgs.append(statistic.hazard_dmg)
                    seed_socs.append(statistic.soc)
                    seed_makespans.append(statistic.makespan)
                seed_fins.append(statistic.fin)

            # add collected statistics from seeds to total statistics
            # we only use successful episodes (nanmean, nanmax, nanmin)
            tmp_mean_seed_hazard_dmgs = (np.nan if not seed_hazard_dmgs
                                         else mean(seed_hazard_dmgs))
            tmp_max_seed_hazard_dmgs = (np.nan if not seed_hazard_dmgs
                                        else max(seed_hazard_dmgs))
            tmp_min_seed_hazard_dmgs = (np.nan if not seed_hazard_dmgs
                                        else min(seed_hazard_dmgs))
            data[Metric.TOTAL_HAZARD_DMG]["avg"][t].append(tmp_mean_seed_hazard_dmgs)
            data[Metric.TOTAL_HAZARD_DMG]["max"][t].append(tmp_max_seed_hazard_dmgs)
            data[Metric.TOTAL_HAZARD_DMG]["min"][t].append(tmp_min_seed_hazard_dmgs)

            tmp_mean_seed_socs = np.nan if not seed_socs else mean(seed_socs)
            tmp_max_seed_socs = np.nan if not seed_socs else max(seed_socs)
            tmp_min_seed_socs = np.nan if not seed_socs else min(seed_socs)
            data[Metric.SOC]["avg"][t].append(tmp_mean_seed_socs)
            data[Metric.SOC]["max"][t].append(tmp_max_seed_socs)
            data[Metric.SOC]["min"][t].append(tmp_min_seed_socs)

            data[Metric.SOC_AND_TOTAL_HAZARD_DMG]["avg"][t].append(tmp_mean_seed_hazard_dmgs+
                                                                   tmp_mean_seed_socs)
            data[Metric.SOC_AND_TOTAL_HAZARD_DMG]["max"][t].append(tmp_max_seed_hazard_dmgs+
                                                                   tmp_max_seed_socs)
            data[Metric.SOC_AND_TOTAL_HAZARD_DMG]["min"][t].append(tmp_min_seed_hazard_dmgs+
                                                                   tmp_min_seed_socs)

            data[Metric.MAKESPAN]["avg"][t].append((np.nan if not seed_makespans
                                                    else mean(seed_makespans)))
            data[Metric.MAKESPAN]["max"][t].append((np.nan if not seed_makespans
                                                    else max(seed_makespans)))
            data[Metric.MAKESPAN]["min"][t].append((np.nan if not seed_makespans
                                                    else min(seed_makespans)))

            # min, max of success rate should be the same as mean (no shaded area)
            tmp_mean_seed_fins = mean(seed_fins)
            data[Metric.SUCCESS_RATE]["avg"][t].append(tmp_mean_seed_fins)
            data[Metric.SUCCESS_RATE]["max"][t].append(tmp_mean_seed_fins)
            data[Metric.SUCCESS_RATE]["min"][t].append(tmp_mean_seed_fins)

    # now pickle the data
    experiment_dir: str = f"{data_path}map({map_id})_withdecay({with_decay})_" + \
                          f"varparam({varying_param})_dmgtype({dmg_type})"

    # create the experiment directory
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    pickle.dump(data, open(f"{experiment_dir}/data.pkl", 'wb'))

    return experiment_dir
