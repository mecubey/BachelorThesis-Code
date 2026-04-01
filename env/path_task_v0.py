import numpy as np
from pprint import pprint
from .implementation.path_task import PathTaskEnv

num_agents = 2
num_tasks = 6
obs_radius = 1
step_rwd = -0.1
task_exc_time_limits = [1, 3]
task_rwd_limits = [5, 10]
maze_intensity = 1
with_zone = True
max_num_spread = 3
step_spread_prob = 0.6
spread_probs = [0.5, 0.5, 0.5, 0.5]
zone_dmg = -1
trait_dim = 5
episode_length = 150
field_dim = 7

def raw_env(render_mode = None, delay_btw_frames = 0, with_task_infos = False):
    return PathTaskEnv(num_agents, num_tasks, obs_radius, step_rwd,
                       task_exc_time_limits, task_rwd_limits,
                       maze_intensity, with_zone, max_num_spread, step_spread_prob, spread_probs, zone_dmg, 
                       trait_dim, episode_length, field_dim, 
                       render_mode, delay_btw_frames, with_task_infos)

if __name__ == "__main__":
    env = raw_env(render_mode="human", delay_btw_frames=0.5, with_task_infos=True)
    rng = np.random.default_rng(1)
    num_episodes = 1

    for eps in range(num_episodes):
        observations, infos = env.reset(eps)

        while env.agents:
            actions = {agent: rng.choice(np.nonzero(observations[agent]["action_mask"])[0]) for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)

        env.close()