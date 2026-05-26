"""
Utility methods, constants, etc. for experiments.
"""

import sys
sys.path.insert(0, '')

from enum import Enum
from env.implementation.pibt.pibt import PIBT
from env.implementation.header import (EnvParams,
                                       HazardDamageType,
                                       Statistic)
from env.implementation.pregenerate import all_free_tiles, wall_maps

default_params = EnvParams(consider_hazards=True,
                           with_decay=False,
                           hazard_dmg_type=HazardDamageType.CONSTANT,
                           num_agents=200,
                           spawn_prob=0.7,
                           spread_prob=0.4,
                           max_num_spread=10,
                           dir_spread_probs=[0.8, 0.8, 0.8, 0.8],
                           max_timestep=150,
                           render_mode=None)

class PlannerType(Enum):
    """
    Specifies planner type.
    """
    HAZARD_AWARE = 1
    HAZARD_UNAWARE = 2

class HazardParameter(Enum):
    """
    Specifies varying parameter.
    """
    SPAWN_PROB = 1
    SPREAD_PROB = 2
    DIR_SPREAD_PROB = 3

LABELS = ["HA", "NHA"]

def run_experiment(map_idx: int, seed: int, env) -> Statistic:
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

def plot_data(*,
              ax,
              xpoints: list[float],
              ha_ypoints: list[float],
              max_ha_ypoints: list[float],
              min_ha_ypoints: list[float],
              nha_ypoints: list[float],
              max_nha_ypoints: list[float],
              min_nha_ypoints: list[float],
              x_axis_title: str,
              y_axis_title: str,
              x_limits: list[int],
              y_limits: list[int]):
    """
    Plot given data.
    Always produces a graph for hazard aware and
    not hazard aware planner.
    """
    ax.plot(xpoints, ha_ypoints, label=LABELS[0])
    ax.fill_between(xpoints,
                    min_ha_ypoints,
                    max_ha_ypoints,
                    color="blue",
                    alpha=0.2)
    ax.plot(xpoints, nha_ypoints, label=LABELS[1])
    ax.fill_between(xpoints,
                    min_nha_ypoints,
                    max_nha_ypoints,
                    color="orange",
                    alpha=0.2)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_xlim([x_limits[0], x_limits[-1]])
    ax.set_ylim([y_limits[0], y_limits[-1]])
    ax.set_xticks(x_limits)
    ax.set_yticks(y_limits)
