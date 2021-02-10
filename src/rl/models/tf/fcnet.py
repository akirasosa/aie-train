import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from rl.models.tf.layers import DenseLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = DenseLayer(dim)
        self.dense2 = DenseLayer(dim)

    def call(self, x, **kwargs):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class FCNet(TFModelV2):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        super(FCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        inputs = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)),),
            name="observations",
        )

        feature = Encoder(256)(inputs)
        logits_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            kernel_initializer=normc_initializer(0.01),
        )(feature)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            # kernel_initializer=normc_initializer(0.01),
            kernel_initializer=normc_initializer(1.),
        )(feature)

        self.base_model = tf.keras.Model(
            inputs=[inputs],
            outputs=[logits_out, value_out],
        )
        self.register_variables(self.base_model.variables)

        self._value_out = None

    @override(ModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        output, self._value_out = self.base_model([input_dict["obs_flat"]])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        output = output + inf_mask

        return output, state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
