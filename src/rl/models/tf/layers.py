import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            dim,
            kernel_initializer=normc_initializer(1.0),
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = tf.keras.activations.tanh(x)
        return x
