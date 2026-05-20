"""
Utility methods, constants, etc. for experiments.
"""

import sys
sys.path.insert(0, '')

import statistics as st
from env.implementation.pibt.pibt import PIBT
from env.path_task_env_v0 import raw_env
from env.implementation.header import EnvParams, HazardDamageType

seeds = list(range(10))
NUM_EPISODES = 1000

default_params = EnvParams(consider_hazards=True,
                           with_decay=True,
                           hazard_dmg_type=HazardDamageType.CONSTANT,
                           num_agents=30,
                           field_dim=8,
                           maze_intensity=0.2,
                           spawn_prob=0.4,
                           spread_prob=0.6,
                           max_num_spread=20,
                           dir_spread_probs=[0.7, 0.7, 0.7, 0.7],
                           max_timestep=100,
                           render_mode=None)

def get_statistics(params: EnvParams,
                   episode_seeds: list[int]) -> tuple[float,
                                                      float,
                                                      float,
                                                      float]:
    """
    For a given set of seeds and enviroment parameters,
    calculate the average SOC, accumlated hazard damage,
    success rate, makespan.

    Args:
        params (EnvParams): Enviroment parameters.
        episode_seeds (list[int]): Set of seeds.

    Returns:
        tuple[float, float, float, float]:
        Average accumulated hazard damage,
        average normalized SOC,
        success rate,
        average makespawns (averaged over sucessfull episodes).
    """
    episode_hazard_dmgs: list[float] = []
    episode_finishes: list[int] = []
    episode_sum_of_costs: list[float] = []
    episode_makespans: list[float] = []
    env = raw_env(params)
    for seed in episode_seeds:
        env.reset(env_seed=seed,
                  maze_seed=seed,
                  zone_seed=seed)

        planner = PIBT(env.grid,
                       env.zone,
                       env.args.with_decay,
                       seed=seed)

        done = False
        while not done:
            actions_dict = planner.step()
            termination, truncation = env.step(actions_dict)
            done = termination or truncation

        normalized_costs: list[float] = []
        hazard_dmgs: list[float] = []
        shortest_paths_lenghts: list[float] = list(env.logger.shortest_path_lengths_buffer.values())
        shortest_paths_sum: float = sum(shortest_paths_lenghts)
        longest_shortest_path: float = max(shortest_paths_lenghts)
        for agent in env.grid.agent_idx:
            normalized_costs.append(sum(env.logger.cost_of_paths_buffer[agent])/
                                    shortest_paths_sum)
            hazard_dmgs.append(sum(env.logger.hazard_dmg_buffer[agent])/
                               longest_shortest_path)
        episode_hazard_dmgs.append(st.mean(hazard_dmgs))
        episode_sum_of_costs.append(st.mean(normalized_costs))
        episode_finishes.append(env.logger.episode_finish)
        if env.logger.makespan != -1:
            episode_makespans.append(env.logger.makespan/longest_shortest_path)
    if not episode_makespans:
        return (st.mean(episode_hazard_dmgs), st.mean(episode_sum_of_costs),
                st.mean(episode_finishes), -1.0)
    return (st.mean(episode_hazard_dmgs), st.mean(episode_sum_of_costs),
            st.mean(episode_finishes), st.mean(episode_makespans))
