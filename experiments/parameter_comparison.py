"""
Experiments involving differing hazard configurations.
"""

import sys
sys.path.insert(0, '')
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from env.path_task_env_v0 import raw_env
from env.implementation.header import HazardDamageType
from experiment_utils import (default_params,
                              run_experiment,
                              LABELS,
                              HazardParameter,
                              PlannerType,
                              plot_data)
from collections import defaultdict
import numpy as np
from statistics import mean
from env.path_task_env_v0 import raw_env

path = os.path.dirname(os.path.abspath(__file__)) + "/results/"

NUM_VALUES = 20
#varying_param: HazardParameter = HazardParameter.SPAWN_PROB
#dmg_type: HazardDamageType = HazardDamageType.CONSTANT
#MAP_IDX: int = 0
seeds = list(range(100))

base_splits: list[float] = [i/(NUM_VALUES-1) for i in range(NUM_VALUES)]

def gen_graph_name(varying_param, dmg_type, map_idx) -> str:
    """
    Generate a name for the graph
    """
    param_name = ""
    dmg_name = ""
    match(varying_param):
        case HazardParameter.SPAWN_PROB:
            param_name = "spawn_prob"
        case HazardParameter.DIR_SPREAD_PROB:
            param_name = "dir_spread_prob"
        case HazardParameter.SPREAD_PROB:
            param_name = "spread_prob"

    match(dmg_type):
        case HazardDamageType.CONSTANT:
            dmg_name = "constant"
        case HazardDamageType.DISTANCE:
            dmg_name = "distance"

    return f"{param_name}_{dmg_name}_map{map_idx}.png"

def get_param_name(varying_param) -> str:
    """
    Get name of parameter.
    """
    match(varying_param):
        case HazardParameter.SPAWN_PROB:
            return "Spawn Probability"
        case HazardParameter.DIR_SPREAD_PROB:
            return "Directional Spread Probabilities"
        case HazardParameter.SPREAD_PROB:
            return "Spread Probability"

def set_param_value(varying_param, new_param, par_val: int) -> None:
    """
    Set parameter value in experiments.
    """
    match(varying_param):
        case HazardParameter.SPAWN_PROB:
            new_param.spawn_prob = base_splits[par_val]
        case HazardParameter.DIR_SPREAD_PROB:
            new_param.dir_spread_probs = [base_splits[par_val]]*4
        case HazardParameter.SPREAD_PROB:
            new_param.spread_prob = base_splits[par_val]

def experiment_and_plot(varying_param, dmg_type, map_idx):
    """
    Experiment and plot man.
    """
    copied_params = deepcopy(default_params)
    env = raw_env(copied_params)

    hazard_dmgs: dict[PlannerType, list[float]] = defaultdict(list)
    max_hazard_dmgs: dict[PlannerType, list[float]] = defaultdict(list)
    min_hazard_dmgs: dict[PlannerType, list[float]] = defaultdict(list)

    socs: dict[PlannerType, list[float]] = defaultdict(list)
    max_socs: dict[PlannerType, list[float]] = defaultdict(list)
    min_socs: dict[PlannerType, list[float]] = defaultdict(list)

    makespans: dict[PlannerType, list[float]] = defaultdict(list)
    max_makespans: dict[PlannerType, list[float]] = defaultdict(list)
    min_makespans: dict[PlannerType, list[float]] = defaultdict(list)

    fins: dict[PlannerType, list[float]] = defaultdict(list)

    for t in PlannerType:
        copied_params.hazard_dmg_type = dmg_type
        if t == PlannerType.HAZARD_UNAWARE:
            # now we collect data for not hazard aware planner
            copied_params.consider_hazards = False
        for i in range(NUM_VALUES):
            set_param_value(varying_param=varying_param,
                            new_param=copied_params,
                            par_val=i)

            seed_hazard_dmgs: list[float] = []
            seed_socs: list[float] = []
            seed_makespans: list[float] = []
            seed_fins: list[float] = []

            for s in seeds:
                statistic = run_experiment(env=env, map_idx=map_idx, seed=s+(i*len(seeds)))

                # only successfull episodes used
                if statistic.fin:
                    seed_fins.append(statistic.fin)
                    seed_hazard_dmgs.append(statistic.hazard_dmg)
                    seed_socs.append(statistic.soc)
                    seed_makespans.append(statistic.makespan)

            fins[t].append(len(seed_fins)/len(seeds))

            if seed_hazard_dmgs:
                hazard_dmgs[t].append(mean(seed_hazard_dmgs))
                max_hazard_dmgs[t].append(max(seed_hazard_dmgs))
                min_hazard_dmgs[t].append(min(seed_hazard_dmgs))
            else:
                hazard_dmgs[t].append(np.nan)
                max_hazard_dmgs[t].append(np.nan)
                min_hazard_dmgs[t].append(np.nan)

            if seed_socs:
                socs[t].append(mean(seed_socs))
                max_socs[t].append(max(seed_socs))
                min_socs[t].append(min(seed_socs))
            else:
                socs[t].append(np.nan)
                max_socs[t].append(np.nan)
                min_socs[t].append(np.nan)

            if seed_makespans:
                makespans[t].append(mean(seed_makespans))
                max_makespans[t].append(max(seed_makespans))
                min_makespans[t].append(min(seed_makespans))
            else:
                makespans[t].append(np.nan)
                max_makespans[t].append(np.nan)
                min_makespans[t].append(np.nan)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.tight_layout(pad=3.0, rect=[0, 0.1, 1, 1])

    # hazard dmg
    plot_data(ax=ax[0],
            xpoints=base_splits,
            ha_ypoints=hazard_dmgs[PlannerType.HAZARD_AWARE],
            max_ha_ypoints=max_hazard_dmgs[PlannerType.HAZARD_AWARE],
            min_ha_ypoints=min_hazard_dmgs[PlannerType.HAZARD_AWARE],
            nha_ypoints=hazard_dmgs[PlannerType.HAZARD_UNAWARE],
            max_nha_ypoints=max_hazard_dmgs[PlannerType.HAZARD_UNAWARE],
            min_nha_ypoints=min_hazard_dmgs[PlannerType.HAZARD_UNAWARE],
            x_axis_title=get_param_name(varying_param),
            y_axis_title="Total Hazard Damage",
            x_limits=[0, 0.5, 1],
            y_limits=[0, 0.5, 1])

    # soc
    plot_data(ax=ax[1],
            xpoints=base_splits,
            ha_ypoints=socs[PlannerType.HAZARD_AWARE],
            max_ha_ypoints=max_socs[PlannerType.HAZARD_AWARE],
            min_ha_ypoints=min_socs[PlannerType.HAZARD_AWARE],
            nha_ypoints=socs[PlannerType.HAZARD_UNAWARE],
            max_nha_ypoints=max_socs[PlannerType.HAZARD_UNAWARE],
            min_nha_ypoints=min_socs[PlannerType.HAZARD_UNAWARE],
            x_axis_title=get_param_name(varying_param),
            y_axis_title="SoC",
            x_limits=[0, 0.5, 1],
            y_limits=[1, 4, 8])

    # makespan
    plot_data(ax=ax[2],
            xpoints=base_splits,
            ha_ypoints=makespans[PlannerType.HAZARD_AWARE],
            max_ha_ypoints=max_makespans[PlannerType.HAZARD_AWARE],
            min_ha_ypoints=min_makespans[PlannerType.HAZARD_AWARE],
            nha_ypoints=makespans[PlannerType.HAZARD_UNAWARE],
            max_nha_ypoints=max_makespans[PlannerType.HAZARD_UNAWARE],
            min_nha_ypoints=min_makespans[PlannerType.HAZARD_UNAWARE],
            x_axis_title=get_param_name(varying_param),
            y_axis_title="Makespan",
            x_limits=[0, 0.5, 1],
            y_limits=[0.75, 1.5, 2.25, 3])

    # success rate
    ax[3].plot(base_splits, fins[PlannerType.HAZARD_AWARE], label=LABELS[0])
    ax[3].plot(base_splits, fins[PlannerType.HAZARD_UNAWARE], label=LABELS[1])
    ax[3].set_xlabel(get_param_name(varying_param))
    ax[3].set_ylabel("Success Rate")
    ax[3].set_xlim([0, 1])
    ax[3].set_ylim([0, 1.05])
    ax[3].set_xticks([0, 0.5, 1])
    ax[3].set_yticks([0, 0.5, 1])

    handles, labels = ax[0].get_legend_handles_labels()

    # figure-level legend
    fig.legend(handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.01),
            fontsize=12,
            fancybox=True,
            ncol=2)

    plt.savefig(path+gen_graph_name(varying_param, dmg_type, map_idx), dpi=300)

experiment_and_plot(HazardParameter.DIR_SPREAD_PROB,
                    HazardDamageType.CONSTANT,
                    0)

experiment_and_plot(HazardParameter.SPAWN_PROB,
                    HazardDamageType.CONSTANT,
                    0)

experiment_and_plot(HazardParameter.SPREAD_PROB,
                    HazardDamageType.CONSTANT,
                    0)

experiment_and_plot(HazardParameter.DIR_SPREAD_PROB,
                    HazardDamageType.DISTANCE,
                    0)

experiment_and_plot(HazardParameter.SPAWN_PROB,
                    HazardDamageType.DISTANCE,
                    0)

experiment_and_plot(HazardParameter.SPREAD_PROB,
                    HazardDamageType.DISTANCE,
                    0)
