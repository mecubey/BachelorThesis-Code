from .implementation.path_task import PathTaskEnv

num_agents = 3
num_tasks = 8
task_exc_time_limits = [1, 3]
task_rwd_limits = [5, 10]
trait_dim = 4
episode_length = 100
field_dim = 7

def raw_env(render_mode = None, delay_btw_frames = 0, with_task_infos = False):
    return PathTaskEnv(num_agents,
                       num_tasks, task_exc_time_limits, task_rwd_limits,
                       trait_dim, episode_length, field_dim, render_mode, delay_btw_frames, with_task_infos)