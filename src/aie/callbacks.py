from typing import Dict, Optional, Sequence

import pandas as pd
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID

from aie.aie_env import AIEEnv


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: MultiAgentEpisode,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        envs: Sequence[AIEEnv] = base_env.get_unwrapped()
        social_metrics = pd.DataFrame([
            e.env.scenario_metrics()
            for e in envs
        ]).mean().to_dict()

        for k, v in social_metrics.items():
            episode.custom_metrics[k] = v

