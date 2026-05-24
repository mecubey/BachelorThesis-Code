"""
Experiments involving differing hazard configurations.
"""

import sys
sys.path.insert(0, '')
import os
import numpy as np
import matplotlib.pyplot as plt
from env.path_task_env_v0 import raw_env
from experiment_utils import (default_params,
                              run_experiment,
                              NUM_VALUES,
                              PlannerType,
                              LABELS)
from collections import defaultdict
from statistics import mean, stdev

path = os.path.dirname(os.path.abspath(__file__)) + "/results"

env = raw_env(default_params)

hazard_dmgs: dict[PlannerType, list[float]] = defaultdict(list)
socs: dict[PlannerType, list[float]] = defaultdict(list)
makespans: dict[PlannerType, list[float]] = defaultdict(list)
fins: dict[PlannerType, list[float]] = defaultdict(list)

for t in PlannerType:
    if t == PlannerType.HAZARD_UNAWARE:
        # now we collect data for not hazard aware planner
        default_params.consider_hazards = False
    for i in range(NUM_VALUES):
        statistic = run_experiment(default_params, i)

        # we only use successful episodes for hzd_dmg, soc, makespan
        if statistic.fin:
            hazard_dmgs[t].append(statistic.hazard_dmg)
            socs[t].append(statistic.soc)
            makespans[t].append(statistic.makespan)

        fins[t].append(statistic.fin)

fig, ax = plt.subplots(1, 3, figsize=(10, 6))
fig.subplots_adjust(wspace=0.3)

def plot_errorbars(pl, values, title):
    """
    Given a statistic, create error bar with avg, min, max.
    """

    means = np.array([mean(values[PlannerType.HAZARD_AWARE]),
                      mean(values[PlannerType.HAZARD_UNAWARE])])

    mins = np.array([min(values[PlannerType.HAZARD_AWARE]),
                     min(values[PlannerType.HAZARD_UNAWARE])])

    maxs = np.array([max(values[PlannerType.HAZARD_AWARE]),
                     max(values[PlannerType.HAZARD_UNAWARE])])

    stds = np.array([stdev(values[PlannerType.HAZARD_AWARE]),
                     stdev(values[PlannerType.HAZARD_UNAWARE])])

    pl.errorbar(LABELS, means, stds, fmt='ok', lw=5)
    pl.errorbar(LABELS, means, [means - mins, maxs - means],
                fmt='.k', ecolor='gray', lw=2)
    pl.set_xlim(-0.8, 1.8)
    pl.set_title(title)

# hazard dmg
plot_errorbars(ax[0], hazard_dmgs, "Total Hazard Damage")

# soc
plot_errorbars(ax[1], socs, "SoC")

# makespans
plot_errorbars(ax[2], makespans, "Makespan")

plt.savefig(path+"/base_comparison.png")

with open(path+"/base_comparison.txt", "w", encoding="utf-8") as f:
    f.write(f"Success rate of HA: {mean(fins[PlannerType.HAZARD_AWARE])}" + "\n" +
            f"Success rate of NHA: {mean(fins[PlannerType.HAZARD_UNAWARE])}" + "\n" +
            f"Averaged over {NUM_VALUES} episodes.")
