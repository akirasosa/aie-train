from typing import Dict, List

import numpy as np
import tensorflow as tf
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from rl.models.tf.layers import DenseLayer


class RNNModel(RecurrentNetwork):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            hiddens_size=128,
            cell_size=128,
    ):
        super(RNNModel, self).__init__(obs_space, action_space, num_outputs,
                                       model_config, name)
        self.cell_size = cell_size

        input_layer = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        dense1 = DenseLayer(hiddens_size)(input_layer)
        dense2 = DenseLayer(hiddens_size)(dense1)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size,
            return_sequences=True,
            return_state=True,
            name="lstm",
        )(
            inputs=dense2,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )
        lstm_out = tf.keras.layers.LayerNormalization()(lstm_out)

        logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="logits",
            kernel_initializer=normc_initializer(0.01),
        )(lstm_out)

        values = tf.keras.layers.Dense(
            1,
            activation=None,
            name="values",
            kernel_initializer=normc_initializer(0.01),
        )(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c],
        )
        self.register_variables(self.rnn_model.variables)
        # self.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        assert seq_lens is not None
        padded_inputs = input_dict["obs_flat"]
        max_seq_len = tf.shape(padded_inputs)[0] // tf.shape(seq_lens)[0]
        output, new_state = self.forward_rnn(
            add_time_dimension(
                padded_inputs,
                max_seq_len=max_seq_len,
                framework="tf",
            ),
            state,
            seq_lens,
        )
        output = tf.reshape(output, [-1, self.num_outputs])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        output = output + inf_mask

        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
