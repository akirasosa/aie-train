import time

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from aie.env_conf import ENV_CONF_DYSTOPIA
from rl.conf import get_base_ppo_conf
from rl.models.tf.fcnet import FCNet


def get_conf():
    return {
        **get_base_ppo_conf(num_workers=10),
        'vf_loss_coeff': 0.1,
        'env_config': ENV_CONF_DYSTOPIA,
    }


def run():
    ModelCatalog.register_custom_model("my_model", FCNet)

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
