import os
import sys
from pprint import pprint
sys.path.insert(0, '')

from env.path_task.path_task_v0 import raw_env
from sb3_contrib import MaskablePPO

load_path = os.path.dirname(os.path.abspath(__file__)) + "/models/path_task_v0"

env = raw_env(render_mode="human", delay_btw_frames=0.5, with_task_infos=True)

model = MaskablePPO.load(load_path)

observations, infos = env.reset(5)
# 1, 2, 3, 4

pprint(infos)

while env.agents:
    actions = {agent: model.predict(observation=observations[agent]["observation"],
                                    action_masks=observations[agent]["action_mask"],
                                    deterministic=True)[0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()