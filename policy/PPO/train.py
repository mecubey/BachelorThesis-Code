import os
import sys
import supersuit as ss
sys.path.insert(0, '')

from env.path_task.path_task_v0 import raw_env
from pettingzoo.utils import parallel_to_aec
from wrappers import SB3ActionMaskWrapper, mask_fn
from feature_extractor import CustomCombinedExtractor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

save_path = os.path.dirname(os.path.abspath(__file__)) + "/models/path_task_v0"

env = raw_env(render_mode=None, delay_btw_frames=0, with_task_infos=False)
env = parallel_to_aec(env)

env = SB3ActionMaskWrapper(env)
env.reset()
env = ActionMasker(env, mask_fn)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = MaskablePPO(
    MaskableMultiInputActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1
)

model.learn(total_timesteps=100_000)

model.save(save_path)

env.close()