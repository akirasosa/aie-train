import time

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from aie.aie_env import OBS_SPACE_AGENT, ACT_SPACE_AGENT
from rl.conf import get_base_ppo_conf
from rl.models.tf.fcnet_lstm import RNNModel


def get_conf():
    return {
        **get_base_ppo_conf(num_workers=10),
        "sgd_minibatch_size": 3000,  # 60 * 200 * 4 / 3000 = 16 steps of (B=60, L=50, dim)

        "lr": 3e-4,

        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
                "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {
                    "model": {
                        "custom_model": "my_model",
                        'max_seq_len': 50,
                    },
                }),
            },
            "policy_mapping_fn": lambda x: 'learned',
        },
    }


def run():
    ModelCatalog.register_custom_model("my_model", RNNModel)

    trainer = ppo.PPOTrainer(config=get_conf())

    t = time.monotonic()
    while True:
        trainer.train()
        checkpoint = trainer.save()
        print(time.monotonic() - t, "checkpoint saved at", checkpoint)
        t = time.monotonic()


if __name__ == '__main__':
    ray.init()
    run()
