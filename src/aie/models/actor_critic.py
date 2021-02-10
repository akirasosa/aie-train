from typing import Dict, List

from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType
from torch import nn as nn

from aie.models.conv import ConvModel
from libs.torch import init_weights


class ActorCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        dim = 128

        self.feature_model_pi = ConvModel(dim)
        self.feature_model_v = ConvModel(dim)
        # self.feature_model_pi = FlatModel(dim)
        # self.feature_model_v = FlatModel(dim)

        self.fc_pi = nn.Sequential(
            nn.Linear(dim, num_outputs),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(dim, 1),
        )
        self.apply(init_weights)

        self._input_dict = None
        self._features = None

    @override(ModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._input_dict = input_dict
        self._features = self.feature_model_pi(input_dict)

        mask = input_dict['obs']['action_mask'].bool()

        logits = self.fc_pi(self._features)
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward first!"
        # x = self.feature_model_v(self._input_dict)
        # x = self.fc_v(x)
        x = self.fc_v(self._features)
        return x.reshape(-1)
