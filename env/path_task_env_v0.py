"""
Contains the actual enviroment creation method with specified parameters.
"""

from pprint import pprint
import numpy as np
from implementation.planner.prioritized_planner import PrioritizedPlanner
from implementation import header as h
from implementation.path_task_env import PathTaskMultiAgentEnv

params = h.EnvParams(
    num_agents=30,
    maze_intensity=0.3,
    spawn_prob=0,
    spread_prob=0,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    episode_length=150,
    field_dim=30,
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

def main():
    """
    Method to test enviroment.
    """
    try:
        NUM_EPISODES = 1

        env = raw_env(params)

        infos = env.reset(seed=1)
        h.print_infos(infos)

        paths = PrioritizedPlanner.plan(grid=env.global_obs,
                                        remaining_time_limit=env.args.episode_length,
                                        start_positions=env.agent_positions,
                                        agents=env.agents,
                                        rng=env.rng)
        done = False
        input()

        for _ in range(NUM_EPISODES):
            while not done:
                action_dict: dict[str, h.Action] = {}
                for a in env.agents:
                    if env.timestep >= len(paths[a.agent_name]):
                        action_dict[a.agent_name] = h.Action.DO_NOTHING
                        continue
                    action_dict[a.agent_name] = paths[a.agent_name][env.timestep]
                terminated, truncated, infos = env.step(action_dict=action_dict)
                done = terminated or truncated
                input()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
