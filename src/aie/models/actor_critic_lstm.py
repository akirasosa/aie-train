from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from aie.models.conv import ConvModel
from libs.torch import init_weights


class ActorCriticLSTM(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        dim = 128
        self.dim = dim

        self.feature_model_pi = ConvModel(dim)
        self.feature_model_v = ConvModel(dim)
        # self.feature_model_pi = FlatModel(dim)
        # self.feature_model_v = FlatModel(dim)

        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.post_lstm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Hardswish(),
        )

        self.fc_pi = nn.Sequential(
            nn.Linear(dim, num_outputs),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(dim, 1),
        )
        self.apply(init_weights)

        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        device = self.feature_model_pi.fc_all[0].weight.device
        h = [
            torch.zeros([self.dim], dtype=torch.float, device=device),
            torch.zeros([self.dim], dtype=torch.float, device=device),
        ]
        return h

    @override(ModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        x = self.feature_model(input_dict)

        flat_inputs = x
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        logits, new_state = self.forward_rnn(inputs, state, seq_lens)
        logits = torch.reshape(logits, [-1, self.num_outputs])

        mask = input_dict['obs']['action_mask'].bool()
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits, new_state

    @override(RecurrentNetwork)
    def forward_rnn(
            self,
            inputs: TensorType,
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features, lstm_state = self.lstm(inputs, [s.unsqueeze(0) for s in state])
        self._features = self.post_lstm(self._features)
        logits = self.fc_pi(self._features)
        return logits, [s.squeeze(0) for s in lstm_state]

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward first!"
        x = self.fc_v(self._features)
        return x.reshape(-1)
