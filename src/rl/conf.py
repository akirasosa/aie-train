from pathlib import Path

from ray.rllib.agents import ppo

from aie.aie_env import AIEEnv, OBS_SPACE_AGENT, ACT_SPACE_AGENT
from aie.callbacks import MyCallbacks

BASE_CONF = {
    "env": AIEEnv,
    "callbacks": MyCallbacks,
    "multiagent": {
        "policies_to_train": ["learned"],
        "policies": {
            "learned": (None, OBS_SPACE_AGENT, ACT_SPACE_AGENT, {
                "model": {
                    "custom_model": "my_model",
                },
            }),
        },
        "policy_mapping_fn": lambda x: 'learned',
    },
    "no_done_at_end": False,
}

BASE_PPO_CONF = {
    **ppo.DEFAULT_CONFIG,
    **BASE_CONF,
}

# OUT_DIR = Path('/mnt/nfs/vfa-red/akirasosa/ray_results/')
OUT_DIR = Path.home() / 'ray_results'


def get_base_ppo_conf(num_workers: int):
    return {
        **BASE_PPO_CONF,

        "num_gpus": 1,
        "num_workers": num_workers,
        "num_gpus_per_worker": 1 / num_workers,
        'num_envs_per_worker': 60 // num_workers,

        "rollout_fragment_length": 200,
        "train_batch_size": 3000,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,

        "vf_loss_coeff": 0.05,
        "clip_param": 0.25,
        "lambda": 0.9,
        "gamma": 0.99,
        "entropy_coeff": 1e-4,
        "lr": 1e-4,
    }
