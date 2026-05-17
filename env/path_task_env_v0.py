"""
Contains the actual enviroment creation method with specified parameters.
"""

from implementation import header as h
from implementation.path_task_env import PathTaskMultiAgentEnv
from implementation.pibt.pibt import PIBT

params = h.EnvParams(
    num_agents=1000,
    maze_intensity=0,
    spawn_prob=1,
    spread_prob=0.7,
    hazard_dmg_type=h.HazardDamageType.DISTANCE,
    max_num_spread=20,
    dir_spread_probs=[0.8, 0.7, 0.55, 0.8],
    max_timestep=300,
    field_dim=32,
    render_mode="human"
)

def raw_env(args: h.EnvParams):
    """
    Method to generate the enviroment with specified
    enviroment arguments. 
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

            planner = PIBT(env.grid, env.zone)

            input()

            done = False
            while not done:
                actions_dict = planner.step()
                termination, truncation = env.step(actions_dict)
                done = termination or truncation
                input()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
