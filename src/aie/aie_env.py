import numpy as np
from ai_economist import foundation
from ai_economist.foundation.base.base_env import BaseEnvironment
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from aie.env_conf import ENV_CONF_DEFAULT

'''
world-map
0: stone
1: wood
2: house
3: water
4: stone (available ??)
5: wood (available ??)
6: wall

world-idx_map
0: house (mine: 1, others: player_idx + 2)
1: player's location (mine: 1, others: player_idx + 2)
'''
OBS_SPACE_AGENT = spaces.Dict({
    'world-map': spaces.Box(0, 1, shape=(7, 11, 11)),
    'world-idx_map': spaces.Box(0, 5, shape=(2, 11, 11)),
    'world-loc-row': spaces.Box(0, 1, shape=(1,)),
    'world-loc-col': spaces.Box(0, 1, shape=(1,)),
    'world-inventory-Coin': spaces.Box(0, np.inf, shape=(1,)),
    'world-inventory-Stone': spaces.Box(0, np.inf, shape=(1,)),
    'world-inventory-Wood': spaces.Box(0, np.inf, shape=(1,)),
    'time': spaces.Box(0, 1, shape=(1, 1)),

    'Build-build_payment': spaces.Box(0, np.inf, shape=(1,)),
    'Build-build_skill': spaces.Box(0, np.inf, shape=(1,)),

    'ContinuousDoubleAuction-market_rate-Stone': spaces.Box(0, np.inf, shape=(1,)),
    'ContinuousDoubleAuction-price_history-Stone': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-available_asks-Stone': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-available_bids-Stone': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-my_asks-Stone': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-my_bids-Stone': spaces.Box(0, np.inf, shape=(11,)),

    'ContinuousDoubleAuction-market_rate-Wood': spaces.Box(0, np.inf, shape=(1,)),
    'ContinuousDoubleAuction-price_history-Wood': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-available_asks-Wood': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-available_bids-Wood': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-my_asks-Wood': spaces.Box(0, np.inf, shape=(11,)),
    'ContinuousDoubleAuction-my_bids-Wood': spaces.Box(0, np.inf, shape=(11,)),

    'Gather-bonus_gather_prob': spaces.Box(0, 1, shape=(1,)),

    'action_mask': spaces.Box(0, 1, shape=(50,)),

})
ACT_SPACE_AGENT = spaces.Discrete(50)


class AIEEnv(MultiAgentEnv):
    def __init__(self, env_config, force_dense_logging: bool = False):
        self.env: BaseEnvironment = foundation.make_env_instance(**{
            **ENV_CONF_DEFAULT,
            **env_config,
        })
        self.observation_space = OBS_SPACE_AGENT
        self.action_space = ACT_SPACE_AGENT
        self.force_dense_logging = force_dense_logging

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset(force_dense_logging=self.force_dense_logging)
        del obs['p']
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        return obs

    def step(self, actions: MultiAgentDict) -> (MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict):
        obs, r, done, info = self.env.step(actions)
        del obs['p'], r['p'], info['p']
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        return obs, r, done, info
