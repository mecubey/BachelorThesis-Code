"""
Experiments involving differing hazard configurations.
"""

import sys
sys.path.insert(0, '')

from env.path_task_env_v0 import raw_env
from experiment_utils import seeds, default_params, NUM_EPISODES, get_statistics
from pprint import pprint

env = raw_env(default_params)

all_spawn_probs = [i/NUM_EPISODES for i in range(NUM_EPISODES)]
all_spread_probs = [i/NUM_EPISODES for i in range(NUM_EPISODES)]
all_max_num_spread = [i/default_params.max_timestep for i in range(default_params.max_timestep)]
all_dir_spread_probs = [[i/NUM_EPISODES,
                         i/NUM_EPISODES,
                         i/NUM_EPISODES,
                         i/NUM_EPISODES] for i in range(NUM_EPISODES)]


pprint(get_statistics(default_params, seeds))
