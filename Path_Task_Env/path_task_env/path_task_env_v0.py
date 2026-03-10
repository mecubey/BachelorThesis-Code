from env.path_task_env import PathTaskEnv
num_agents = 1
agent_traits = [[1]]
num_tasks = 9
task_requirements = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
task_exc_time_limits = [1, 1]
task_rwd_limits = [0, 10]
episode_length = 10000
field_dims = 3
render_mode = "human"
delay_btw_frames = 0.1

env = PathTaskEnv(num_agents, agent_traits, 
                  num_tasks, task_requirements, task_exc_time_limits, task_rwd_limits,
                  episode_length, field_dims, render_mode, delay_btw_frames)
env.reset()

if __name__ == "__main__":
    observations, infos = env.reset(1)

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()