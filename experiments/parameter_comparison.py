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

for t in HazardDamageType:
    for p in Parameter:
        exp_dir = gen_data(map_id=MAP_ID,
                           varying_param=p,
                           dmg_type=t)

        data: Data = pickle.load(open(f"{exp_dir}/data.pkl", "rb"))

        plot_data(data=data,
                  map_id=MAP_ID,
                  varied_param=p,
                  dmg_type=t)
