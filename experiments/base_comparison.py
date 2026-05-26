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
                              PlannerType,
                              LABELS)
from collections import defaultdict
from statistics import mean, stdev
from env.path_task_env_v0 import raw_env

NUM_VALUES = 2000

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
        statistic = run_experiment(map_idx=0, env=env, seed=i)

        # we only use successful episodes for hzd_dmg, soc, makespan
        if statistic.fin:
            hazard_dmgs[t].append(statistic.hazard_dmg)
            socs[t].append(statistic.soc)
            makespans[t].append(statistic.makespan)

        fins[t].append(statistic.fin)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
fig.tight_layout(pad=3.0, rect=[0, 0, 1, 1])

#fig.subplots_adjust(wspace=0.3)

def plot_errorbars(pl, values, title, y_limits):
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

    pl.errorbar(LABELS[0], means[0], stds[0], fmt='ok', lw=5, ecolor="blue")
    pl.errorbar(LABELS[0], means[0], [[means[0] - mins[0]], [maxs[0] - means[0]]],
                fmt='.k', ecolor='gray', lw=2)
    pl.errorbar(LABELS[1], means[1], stds[1], fmt='ok', lw=5, ecolor="orange")
    pl.errorbar(LABELS[1], means[1], [[means[1] - mins[1]], [maxs[1] - means[1]]],
                fmt='.k', ecolor='gray', lw=2)
    pl.set_xlim(-0.8, 1.8)
    pl.set_xticks([])
    pl.set_ylim([y_limits[0], y_limits[-1]])
    pl.set_yticks(y_limits)
    pl.set_ylabel(title)

# hazard dmg
plot_errorbars(ax[0], hazard_dmgs, "Total Hazard Damage", [0, 0.5, 1])

# soc
plot_errorbars(ax[1], socs, "SoC", [1, 4, 8])

# makespans
plot_errorbars(ax[2], makespans, "Makespan", [0.75, 1.5, 2.25, 3])

from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color='blue', lw=1, label='HA'),
    Line2D([0], [0], color='orange', lw=1, label='NHA')
]

# figure-level legend
fig.legend(handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        fontsize=12,
        fancybox=True,
        ncol=2)

plt.savefig(path+"/base_comparison.png")

with open(path+"/base_comparison.txt", "w", encoding="utf-8") as f:
    f.write(f"Success rate of HA: {mean(fins[PlannerType.HAZARD_AWARE])}" + "\n" +
            f"Success rate of NHA: {mean(fins[PlannerType.HAZARD_UNAWARE])}" + "\n" +
            f"Averaged over {NUM_VALUES} episodes.")
