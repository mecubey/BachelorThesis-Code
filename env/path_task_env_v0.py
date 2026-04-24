"""
Contains the actual enviroment creation method with specified parameters.
"""

from .implementation import header as h
from .implementation.path_task_env import PathTaskEnv

params = h.EnvParams(
    num_agents=2,
    num_tasks=2,
    obs_radius=2,
    agent_capability=0.3,
    maze_intensity=0,
    step_spread_prob=0,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    trait_dim=5,
    episode_length=150,
    field_dim=3,
    render_mode=None,
    delay_btw_frames=0,
    with_debug_infos=False
)

def raw_env():
    """
    It is preferred to use this method for enviroment creation and change
    the parameters in the file of this method.
    """
    return PathTaskEnv(args=params)
