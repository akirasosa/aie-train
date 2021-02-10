import gym
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from rl.models.const import KEYS_3D
from rl.models.tf.layers import DenseLayer


class DenseLayers(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = DenseLayer(dim)
        self.dense2 = DenseLayer(dim)

    def call(self, x, **kwargs):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class Conv2DLayers(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            dim,
            3,
            strides=(1, 1),
            padding="valid",
            activation='relu',
        )
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(
            dim,
            3,
            strides=(1, 1),
            padding="valid",
            activation='relu',
        )
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = DenseLayer(dim)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2DLayers(dim)
        self.dense = DenseLayers(dim)

    def call(self, x, **kwargs):
        x_conv, x_dense = x
        x_conv = self.conv(x_conv)
        x_dense = self.dense(x_dense)
        x = x_conv + x_dense
        return x


class ConvFCNet(TFModelV2):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        super(ConvFCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        inputs_conv = tf.keras.layers.Input(
            shape=(11, 11, 9),
        )
        inputs_dense = tf.keras.layers.Input(
            shape=(1260 - 11 * 11 * 9,),
        )
        feats = Encoder(128)((inputs_conv, inputs_dense))

        logits_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            kernel_initializer=normc_initializer(0.01),
        )(feats)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            kernel_initializer=normc_initializer(0.01),
        )(feats)

        self.base_model = tf.keras.Model(
            inputs=[inputs_conv, inputs_dense],
            outputs=[logits_out, value_out],
        )
        print(self.base_model.summary())
        self.register_variables(self.base_model.variables)

        self._value_out = None

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # inputs_conv = tf.concat([input_dict['obs'][k] for k in KEYS_3D], axis=1)
        inputs_conv = tf.concat([input_dict['obs'][k] for k in KEYS_3D], axis=1)
        inputs_conv = tf.transpose(inputs_conv, perm=(0, 2, 3, 1))

        inputs_dense = input_dict["obs_flat"][:, (9 * 11 * 11):]

        output, self._value_out = self.base_model([
            inputs_conv,
            inputs_dense,
        ])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        output = output + inf_mask

        return output, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
