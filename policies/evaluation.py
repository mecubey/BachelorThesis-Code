
import sys
sys.path.insert(0, '')

import torch
import numpy as np
import env.implementation.header as h
from env.path_task_env_v0 import raw_env
import os
from ray.rllib.core.rl_module import RLModule

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = path_to_this_dir + "/checkpoints"

policy_path = "/learner_group/learner/rl_module/shared_policy"

rl_module = RLModule.from_checkpoint(checkpoint_dir + "/checkpoint_1000" + policy_path)

env = raw_env()


# Reset the env to get the initial observation.
obs, info = env.reset(seed=2)

input()
done = False

episode_return = 0

while not done:
    action_dict = {}

    for a in env.agents:
        # Compute the next action from a batch (B=1) of observations.
        grid_obs = torch.from_numpy(obs[a]["local_obs"]["grid_obs"]).unsqueeze(0) # add batch B=1 dimension
        vec_obs = torch.from_numpy(obs[a]["local_obs"]["vec_obs"]).unsqueeze(0) # add batch B=1 dimension
        mask = torch.from_numpy(obs[a]["action_mask"]).unsqueeze(0) # add batch B=1 dimension
        model_outputs = rl_module.forward_inference({"obs": {"local_obs": 
                                                             {"grid_obs": grid_obs,
                                                              "vec_obs": vec_obs},
                                                             "action_mask": mask}})
        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
        
        # TODO: ACTION MASKING

        # For discrete actions, you should take the argmax over the logits:
        greedy_action = np.argmax(action_dist_params)

        action_dict[a] = greedy_action

    # Send the action to the environment for the next step.
    obs, rewards, terminateds, truncateds, info = env.step(action_dict)
    print(rewards)
    input()

    # Perform env-loop bookkeeping.
    episode_return += np.average(list(rewards.values()))
    done = terminateds["__all__"] or truncateds["__all__"]

print(f"Reached average episode reward return per agent of {episode_return}.")