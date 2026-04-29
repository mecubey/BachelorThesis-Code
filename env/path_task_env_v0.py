"""
Contains the actual enviroment creation method with specified parameters.
"""

from pprint import pprint
import numpy as np
from implementation import header as h
from implementation.path_task_env import PathTaskMultiAgentEnv

params = h.EnvParams(
    num_agents=2,
    num_tasks=2,
    obs_radius=2,
    agent_capability=0.3,
    maze_intensity=0,
    spawn_prob=0,
    spread_prob=0,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    trait_dim=5,
    episode_length=150,
    field_dim=2,
    render_mode="human",
    delay_btw_frames=0,
    with_debug_infos=True
)

def raw_env(args: h.EnvParams):
    """
    It is preferred to use this method for enviroment creation and change
    the parameters in the file of this method.
    """
    return PathTaskMultiAgentEnv(args=args)

if __name__ == "__main__":
    try:
        NUM_EPISODES = 1

        env = raw_env(params)

        obs, infos = env.reset(seed=1)
        input()

        done = False

        for _ in range(NUM_EPISODES):
            while not done:
                action_dict: dict[str, h.Action] = {}
                for a in env.agents:
                    mask: h.FloatArr = obs[a.agent_name]["action_mask"] # type: ignore
                    action_dict[a.agent_name] = np.random.choice(np.nonzero(mask)[0])
                obs, rwds, term, trunc, infos = env.step(action_dict=action_dict)
                input()
                done = term or trunc
    except KeyboardInterrupt:
        pass
