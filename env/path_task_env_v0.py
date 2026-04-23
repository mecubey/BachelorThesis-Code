"""
Contains the actual enviroment creation method with specified parameters.
"""

from pprint import pprint
import numpy as np
import implementation.header as h
from implementation.path_task_env import PathTaskEnv

params = h.EnvParams(
    num_agents=7,
    num_tasks=10,
    obs_radius=2,
    agent_capability=0.3,
    maze_intensity=0.7,
    step_spread_prob=0.6,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    trait_dim=5,
    episode_length=150,
    field_dim=10,
    render_mode="human",
    delay_btw_frames=0,
    with_debug_infos=True
)

def raw_env():
    """
    It is preferred to use this method for enviroment creation and change
    the parameters in the file of this method.
    """
    return PathTaskEnv(args=params)

if __name__ == "__main__":
    env = raw_env()
    NUM_EPISODES = 1

    rng = np.random.default_rng(seed=1)

    try:
        for _ in range(NUM_EPISODES):
            observations, infos = env.reset(seed=3)
            done: bool = False
            input()

            while not done:
                actions = {agent: rng.choice(np.nonzero(observations[agent]["action_mask"])[0])
                        for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                done = terminations["__all__"] or truncations["__all__"]
                input()

            env.close()
    except KeyboardInterrupt:
        pass
