import sys
sys.path.insert(0, '')

from env.path_task_env_v0 import raw_env
from policy_rl_module import CentralizedCriticSharedActor
from multi_rl_module import CentralizedCriticSharedActorMultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

def env_creator(config):
    return raw_env()

register_env("path_task", env_creator)

env = env_creator(None) # to get obs & action space

config = (
    PPOConfig()
    .environment("path_task")
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, **kw: f"shared_policy"
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            multi_rl_module_class=CentralizedCriticSharedActorMultiRLModule,
            rl_module_specs={
                "p0": RLModuleSpec(
                    module_class=CentralizedCriticSharedActor,
                    observation_space=env.get_observation_space(),
                    action_space=env.get_action_space(),
                    model_config={},
                    catalog_class=None
                )
            }
        )
    )
    .env_runners(num_env_runners=1)
)

config.training()

algo = config.build_algo()

