import sys
import os
sys.path.insert(0, '')

from env.path_task_env.path_task_env_v0 import env
from pettingzoo.utils import parallel_to_aec
from policy.wrappers import train, eval

model_path = os.path.dirname(os.path.abspath(__file__)) + "/models/SB3_PPO_path_task_env"
# parameters of currently saved model:
#  num_agents = 2
#  num_tasks = 6
#  task_exc_time_limits = [1, 3]
#  task_rwd_limits = [5, 10]
#  trait_dim = 6
#  episode_length = 50
#  field_dim = 3



#train_path_task_env = env(render_mode=None, delay_btw_frames=0, with_task_infos=False)
#train_path_task_env = parallel_to_aec(train_path_task_env)
#train(train_path_task_env, model_path, steps=1_000_000, seed=None)

#eval_path_task_env = env(render_mode=None, delay_btw_frames=0, with_task_infos=True)
#eval(eval_path_task_env, model_path, seed=None)