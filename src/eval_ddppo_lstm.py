import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ddppo
from ray.rllib.models import ModelCatalog

from aie import plotting
from aie.aie_env import AIEEnv
from aie.models.actor_critic_lstm import ActorCriticLSTM
from train_ddppo_lstm import get_conf

# %%
ray.init()
ModelCatalog.register_custom_model("my_torch_model", ActorCriticLSTM)

# %%
trainer = ddppo.DDPPOTrainer(env=AIEEnv, config={
    **get_conf(),
    "num_workers": 1,
})
trainer.restore('/home/akirasosa/ray_results/DDPPO_AIEEnv_2021-02-09_13-51-39xi97rawu/checkpoint_159/checkpoint-159')

# %%
env = AIEEnv({}, force_dense_logging=True)
obs = env.reset()
plot_every = 1000
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

hidden_states = {
    k: [
        torch.zeros([128], dtype=torch.float, device='cuda'),
        torch.zeros([128], dtype=torch.float, device='cuda'),
    ]
    for k in obs.keys()
}
obs_list = []
for t in range(1000):
    results = {
        k: trainer.compute_action(
            v,
            state=hidden_states[k],
            policy_id='learned',
            explore=False,
        )
        for k, v in obs.items()
    }
    actions = {
        k: v[0]
        for k, v in results.items()
    }
    hidden_states = {
        k: v[1]
        for k, v in results.items()
    }
    obs, reward, done, info = env.step(actions)
    obs_list.append(obs)

    if ((t + 1) % plot_every) == 0:
        plotting.plot_env_state(env.env, ax)
        ax.set_aspect('equal')
        fig.show()

if ((t + 1) % plot_every) != 0:
    plotting.plot_env_state(env.env, ax)
    ax.set_aspect('equal')
    fig.show()

# %%
dense_log = env.env.previous_episode_dense_log

# %%
fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)
fig.show()

# %%
plotting.breakdown(dense_log)
plt.show()

# %%
for p in map(str, range(4)):
    plt.plot(range(1000), np.cumsum([
        r[p]
        for r in dense_log['rewards']
    ]))
plt.legend(range(4))
plt.show()
plt.close()

# %%
stacked = [
    {
        k: np.stack([
            o[str(n)][k]
            for o in obs_list
        ])
        for k in obs['0'].keys()
    }
    for n in range(4)
]
stacked = {
    k: np.concatenate([
        s[k]
        for s in stacked
    ])
    for k in stacked[0].keys()
}
for i, (k, v) in enumerate(stacked.items()):
    print(k, v.mean(), v.std(), v.min(), v.max())
    plt.hist(v.reshape(-1), bins=100)
    plt.title(k)
    plt.savefig(f'/tmp/{i}.png')
    # plt.show()
    plt.close()
