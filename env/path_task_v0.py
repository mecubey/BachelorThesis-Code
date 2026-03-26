import numpy as np
from pprint import pprint
from .implementation.path_task import PathTaskEnv

num_agents = 1
num_tasks = 2
task_exc_time_limits = [1, 3]
task_rwd_limits = [5, 10]
maze_intensity = 0
with_zone = False
max_num_spread = 3
step_spread_prob = 0.6
spread_probs = [0.75, 0.75, 0.75, 0.75]
zone_dmg = -1
trait_dim = 5
episode_length = 150
field_dim = 10

def raw_env(render_mode = None, delay_btw_frames = 0, with_task_infos = False):
    return PathTaskEnv(num_agents,
                       num_tasks, task_exc_time_limits, task_rwd_limits,
                       maze_intensity, with_zone, max_num_spread, step_spread_prob, spread_probs, zone_dmg, 
                       trait_dim, episode_length, field_dim, 
                       render_mode, delay_btw_frames, with_task_infos)

if __name__ == "__main__":
    env = raw_env(render_mode=None, delay_btw_frames=0, with_task_infos=True)
    observations, infos = env.reset()
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: np.random.choice(np.nonzero(observations[agent]["action_mask"])[0]) for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()