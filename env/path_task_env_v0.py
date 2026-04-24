"""
Contains the actual enviroment creation method with specified parameters.
"""

import implementation.header as h

params = h.EnvParams(
    num_agents=1,
    num_tasks=1,
    obs_radius=2,
    agent_capability=0,
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
