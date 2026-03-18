import numpy as np
from .implementation.path_task_env import PathTaskEnv

num_agents = 2
num_tasks = 6
task_exc_time_limits = [1, 3]
task_rwd_limits = [5, 10]
trait_dim = 6
episode_length = 50
field_dim = 3

def env(render_mode = None, delay_btw_frames = 0, with_task_infos = False):
    return PathTaskEnv(num_agents,
                       num_tasks, task_exc_time_limits, task_rwd_limits,
                       trait_dim, episode_length, field_dim, render_mode, delay_btw_frames, with_task_infos)

if __name__ == "__main__":
    path_task_env = env("human", 1, True)

    observations, infos = path_task_env.reset()
    
    while path_task_env.agents:
        actions = {agent: np.random.choice(observations[agent]["action_mask"].nonzero()[0]) for agent in path_task_env.agents}
        
        observations, rewards, terminations, truncations, infos = path_task_env.step(actions)

    path_task_env.close()