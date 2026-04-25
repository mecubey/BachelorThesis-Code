from typing import Dict, Any, Union, override
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.utils.typing import ModuleID

class CentralizedCriticSharedActorMultiRLModule(MultiRLModule):
    def setup(self):
        super().setup()

    def _forward(self, batch, forward_type, **kwargs):
        m = getattr(self._rl_modules["shared_policy"], forward_type)
        fwd_out = {}
        fwd_out["shared_policy"] = m(batch["shared_policy"], **kwargs)
        return fwd_out
