from path_task_env import PathTaskEnv
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = PathTaskEnv(4, 2, 4, 100, [10, 10])
    parallel_api_test(env, num_cycles=1000)