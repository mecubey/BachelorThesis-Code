"""
Contains the actual enviroment creation method with specified parameters.
"""

from implementation.planner.prioritized_planner import PrioritizedPlanner
from implementation import header as h
from implementation.path_task_env import PathTaskMultiAgentEnv

params = h.EnvParams(
    num_agents=8,
    maze_intensity=0.5,
    spawn_prob=1,
    spread_prob=1,
    max_num_spread=5,
    dir_spread_probs=[0.6, 0.7, 0.55, 0.8],
    max_timestep=150,
    field_dim=8,
    render_mode="human",
    delay_btw_frames=0
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

    try:
        num_episodes = 1

        env.reset()

        actions = PrioritizedPlanner.plan(remaining_time_limit=env.args.max_timestep,
                                          start_positions=env.agent_positions,
                                          goal_positions=env.goal_positions,
                                          trajectories=[],
                                          rng=env.rng,
                                          no_wall=env.on_no_wall,
                                          inside_grid=env.inside_grid)
        done = False
        input()

        for _ in range(num_episodes):
            while not done:
                action_dict: dict[str, h.Action] = {}
                for i in env.agent_idx:
                    agent_name = f"agent_{i}"
                    if env.timestep >= len(actions[agent_name]):
                        action_dict[agent_name] = h.Action.DO_NOTHING
                        continue
                    action_dict[agent_name] = actions[agent_name][env.timestep]
                terminated, truncated = env.step(action_dict=action_dict)
                done = terminated or truncated
                input()
        env.close()
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    main()
