import ray
from ray.rllib.agents.ppo import ddppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from tqdm import tqdm

from aie.aie_env import AIEEnv, OBS_SPACE_AGENT, ACT_SPACE_AGENT
from aie.callbacks import MyCallbacks
from aie.models.actor_critic import ActorCritic


def get_conf():
    return {
        **ddppo.DEFAULT_CONFIG,
        "env": AIEEnv,
        "num_workers": 8,
        "num_gpus_per_worker": 0.1,

        "callbacks": MyCallbacks,

        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
                "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {
                    "model": {
                        "custom_model": "my_torch_model",
                        "vf_share_layers": True,
                    },
                    "framework": "torch",
                }),
            },
            "policy_mapping_fn": lambda x: 'learned',
        },
        "no_done_at_end": False,
        "framework": "torch",
    }


def run():
    trainer = ddppo.DDPPOTrainer(config=get_conf())

    for _ in tqdm(range(1000)):
        result = trainer.train()
        print(pretty_print(result))

        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)


if __name__ == '__main__':
    ray.init()
    ModelCatalog.register_custom_model("my_torch_model", ActorCritic)

    run()
