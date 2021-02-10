import time

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from rl.conf import get_base_ppo_conf
from rl.models.torch.fcnet import FCNet


def get_conf():
    lr = 1e-4

    return {
        **get_base_ppo_conf(num_workers=10),
        'lr_schedule': [
            [0, lr],
            [10_000_000, lr],
            [15_000_000, 0],
        ],
        'framework': 'torch'
    }


def run():
    ModelCatalog.register_custom_model("my_model", FCNet)

    trainer = ppo.PPOTrainer(config=get_conf())

    t = time.monotonic()
    while True:
        trainer.train()
        checkpoint = trainer.save()
        print(time.monotonic() - t, "checkpoint saved at", checkpoint)
        # print(time.monotonic() - t, "checkpoint saved at")
        t = time.monotonic()


if __name__ == '__main__':
    ray.init()
    run()
