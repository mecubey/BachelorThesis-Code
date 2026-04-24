# needed API: [<class 'ray.rllib.core.rl_module.apis.value_function_api.ValueFunctionAPI'>]

from typing import Any, Dict, Optional
from ray.rllib.utils.typing import TensorType
from cetralized_critic_network import CentralizedCritic
from actor_network import Actor
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

class CentralizedCriticSharedActor(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        super().setup()

        h_global, w_global, c_global = self.observation_space["global_obs"].shape
        h_local, w_local, c_local = self.observation_space["local_obs"]["grid_obs"].shape
        vec_dim = self.observation_space["local_obs"]["vec_obs"].shape[0]
        
        self.vf = CentralizedCritic(h_global, w_global, c_global)
        self.encoder = Actor(h_local, w_local, c_local, vec_dim, self.action_space.n)

    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        return self.vf(batch["global_obs"])

    def _forward(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: self.encoder(batch["local_obs"]["grid_obs"],
                                                       batch["local_obs"]["vec_obs"],
                                                       batch["local_obs"]["action_mask"])}