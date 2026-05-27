"""
Experiments involving differing hazard configurations.
"""

import sys
sys.path.insert(0, '')

import pickle
from env.implementation.pregenerate import generate_maps
from experiment_utils import (Parameter,
                              HazardDamageType,
                              Data)
from generate_data import gen_data
from plot_data import plot_data

generate_maps()

MAP_ID = 0

for with_decay in [False, True]:
    for dmg_type in HazardDamageType:
        for param in Parameter:
            if (not with_decay and
                dmg_type == HazardDamageType.CONSTANT and
                param == Parameter.SPAWN_PROB):
                continue
            exp_dir = gen_data(map_id=MAP_ID,
                               with_decay=with_decay,
                               varying_param=param,
                               dmg_type=dmg_type)

            data: Data = pickle.load(open(f"{exp_dir}/data.pkl", "rb"))

            plot_data(data=data,
                      with_decay=with_decay,
                      map_id=MAP_ID,
                      varied_param=param,
                      dmg_type=dmg_type)
