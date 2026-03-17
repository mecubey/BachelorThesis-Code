from pettingzoo.test import parallel_api_test

import sys
sys.path.insert(0, '')

from env.path_task_env.path_task_env_v0 import env

if __name__ == "__main__":
    path_task_env = env(render_mode=None, delay_btw_frames=0, with_task_infos=False)
    parallel_api_test(path_task_env, num_cycles=1_000_000)