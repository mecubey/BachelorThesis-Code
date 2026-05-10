"""
Contains the actual enviroment creation method with specified parameters.
"""

from implementation import header as h
from implementation.planner.prioritized_planner import PrioritizedPlanner
from implementation.path_task_env import PathTaskMultiAgentEnv

params = h.EnvParams(
    num_agents=20,
    maze_intensity=0.5,
    spawn_prob=1,
    spread_prob=1,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    max_timestep=150,
    field_dim=12,
    render_mode="human"
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
    env = raw_env(params)
    num_episodes = 1

    try:
        for _ in range(num_episodes):
            env.reset(env_seed=1,
                      maze_seed=1,
                      zone_seed=1)

            planner = PrioritizedPlanner(env.grid,
                                         env.args.max_timestep,
                                         seed=1)

            planner.initial_plan()
            input()

            done = False
            while not done:
                action_dict: dict[str, h.Action] = planner.get_actions_at(env.timestep)
                terminated, truncated = env.step(action_dict=action_dict)
                done = terminated or truncated
                input()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
