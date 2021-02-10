from typing import Dict, List, Callable, Any

import torch
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType
from torch import nn as nn
from torch.nn import init


def normc_initializer(std: float = 1.0) -> Callable[[nn.Module], None]:
    def initializer(m: nn.Module):
        m.weight.data.normal_(0, 1)
        m.weight.data *= std / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            init.zeros_(m.bias)

    return initializer


class LinearWithInit(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            initializer: Callable[[nn.Module], None],
    ):
        super().__init__(in_features, out_features)
        self.apply(initializer)


class Encoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        input_dim = 1260
        self.dense = nn.Sequential(
            LinearWithInit(input_dim, dim, normc_initializer(1.)),
            nn.LayerNorm(dim),
            nn.Tanh(),
            LinearWithInit(dim, dim, normc_initializer(1.)),
            nn.LayerNorm(dim),
            nn.Tanh(),
        )

    def forward(self, input_dict: Dict[str, TensorType]):
        x = self.dense(input_dict["obs_flat"])
        return x


class FCNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        dim = 256

        self.encoder = Encoder(dim)
        self.fc_pi = LinearWithInit(dim, num_outputs, normc_initializer(0.01))
        self.fc_v = LinearWithInit(dim, 1, normc_initializer(1.))

        self._features = None

    @override(ModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = self.encoder(input_dict)

        mask = input_dict['obs']['action_mask'].bool()

        logits = self.fc_pi(self._features)
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward first!"
        x = self.fc_v(self._features)
        return x.reshape(-1)
