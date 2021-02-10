import matplotlib.pyplot as plt
import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from tqdm import tqdm

from aie import plotting
from aie.aie_env import AIEEnv
from rl.conf import BASE_PPO_CONF, OUT_DIR
from rl.models.tf.fcnet import FCNet

# %%
ray.init()
ModelCatalog.register_custom_model("my_model", FCNet)

# %%
trainer = ppo.PPOTrainer(config={
    **BASE_PPO_CONF,
    "num_workers": 0,
})

ckpt_path = OUT_DIR / 'PPO_AIEEnv_2021-02-19_16-19-44p6xanojq/checkpoint_1777/checkpoint-1777'

trainer.restore(str(ckpt_path))

# %%
env = AIEEnv({}, force_dense_logging=True)
obs = env.reset()

for t in tqdm(range(1000)):
    results = {
        k: trainer.compute_action(
            v,
            policy_id='learned',
            explore=False,
        )
        for k, v in obs.items()
    }
    obs, reward, done, info = env.step(results)

# %%
plotting.breakdown(env.env.previous_episode_dense_log)
plt.show()

# %%
env.env.scenario_metrics()
