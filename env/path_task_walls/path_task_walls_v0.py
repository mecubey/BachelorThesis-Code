from .implementation.path_task_walls import PathTaskWallsEnv

num_agents = 3
num_tasks = 8
task_exc_time_limits = [1, 5]
task_rwd_limits = [5, 10]
maze_intensity = 0.8
trait_dim = 5
episode_length = 150
field_dim = 8

def raw_env(render_mode = None, delay_btw_frames = 0, with_task_infos = False):
    return PathTaskWallsEnv(num_agents,
                            num_tasks, task_exc_time_limits, task_rwd_limits,
                            maze_intensity, trait_dim, episode_length, field_dim, render_mode, delay_btw_frames, with_task_infos)