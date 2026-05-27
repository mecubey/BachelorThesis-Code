"""
Utility methods, constants, etc. for experiments.
"""

import sys
sys.path.insert(0, '')

from enum import StrEnum
from env.implementation.header import (EnvParams,
                                       HazardDamageType)

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

LABELS = ["HA", "HUA"]

SEEDS = list(range(50))
NUM_SPLITS = 20
BASE_SPLITS = [i/(NUM_SPLITS-1) for i in range(NUM_SPLITS)]
DIR_SPREAD_SPLITS = [[elem]*4 for elem in BASE_SPLITS]
AGENTS_SPLITS = [int(300*elem) for elem in BASE_SPLITS]

class PlannerType(StrEnum):
    """
    Specifies planner type.
    """
    HAZARD_AWARE = "HA"
    HAZARD_UNAWARE = "HUA"

class Parameter(StrEnum):
    """
    Specifies varying parameter.
    """
    SPAWN_PROB = "spawn probability"
    SPREAD_PROB = "spread probability"
    DIR_SPREAD_PROB = "directional spread probabilities"
    AGENTS = "agents"

class Metric(StrEnum):
    """
    Specifies recorded metric.
    """
    TOTAL_HAZARD_DMG = "total hazard damage"
    SOC = "soc"
    SOC_AND_TOTAL_HAZARD_DMG = "soc + total hazard damage"
    MAKESPAN = "makespan"
    SUCCESS_RATE = "success rate"

METRIC_AXES = {Metric.TOTAL_HAZARD_DMG: (1, 0),
               Metric.SOC: (1, 1),
               Metric.SUCCESS_RATE: (0, 2),
               Metric.MAKESPAN: (0, 1),
               Metric.SOC_AND_TOTAL_HAZARD_DMG: (1, 2)}

PARAMETER_LIMITS = {Parameter.SPAWN_PROB: [0, 0.5, 1],
                    Parameter.SPREAD_PROB: [0, 0.5, 1],
                    Parameter.DIR_SPREAD_PROB: [0, 0.5, 1],
                    Parameter.AGENTS: [0, 50, 100, 150, 200, 250, 300]}

METRIC_LIMITS = {Metric.TOTAL_HAZARD_DMG: [0, 1, 2, 3, 4],
                 Metric.SOC: [0, 1, 2, 3, 4],
                 Metric.SUCCESS_RATE: [0, 0.5, 1.05],
                 Metric.MAKESPAN: [1, 2, 3, 4],
                 Metric.SOC_AND_TOTAL_HAZARD_DMG: [0, 1, 2, 3, 4]}

Data = dict[Metric, dict[str, dict[PlannerType, list[float]]]]

def get_graph_name(*,
                   with_decay: bool,
                   varying_param: Parameter,
                   dmg_type: HazardDamageType,
                   map_idx: int) -> str:
    """
    Generate a name for the plot of the experiment.
    """
    return f"with_delay({with_decay})_{varying_param}_{dmg_type}_map{map_idx}.png"
