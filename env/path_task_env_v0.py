"""
Contains the actual enviroment creation method with specified parameters.
"""

from implementation import header as h
from implementation.path_task_env import PathTaskMultiAgentEnv
from implementation.pibt_plus.pibt_plus import PIBTPLUS

params = h.EnvParams(
    num_agents=13,
    maze_intensity=1,
    spawn_prob=0,
    spread_prob=0.3,
    max_num_spread=20,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    max_timestep=200,
    field_dim=8,
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
            env.reset(env_seed=3,
                      maze_seed=6,
                      zone_seed=10)

            planner = PIBTPLUS(env.grid, pibt_seed=0, lacam_seed=0)

            input()

            done = False
            while not done:
                print(planner.t_min-env.timestep)
                actions_dict = planner.step(env.timestep)
                termination, truncation = env.step(actions_dict)
                done = termination or truncation
                input()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
