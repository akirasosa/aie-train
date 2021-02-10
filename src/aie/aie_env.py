import numpy as np
from ai_economist import foundation
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import LayoutFromFile
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

ENV_CONF_DEFAULT = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'layout_from_file/simple_wood_and_stone',

    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #   {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
    ],

    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,

    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,  # Number of non-planner agents (must be > 1)
    'world_size': [25, 25],  # [Height, Width] of the env world
    'episode_length': 1000,  # Number of timesteps per episode

    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,

    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': False,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
}

_unbound = [float('-inf'), float('inf')]
OBS_SPACE_AGENT = spaces.Dict({
    'world-map': spaces.Box(*_unbound, shape=(7, 11, 11)),
    'world-idx_map': spaces.Box(*_unbound, shape=(2, 11, 11)),
    'world-loc-row': spaces.Box(*_unbound, shape=(1,)),
    'world-loc-col': spaces.Box(*_unbound, shape=(1,)),
    'world-inventory-Coin': spaces.Box(*_unbound, shape=(1,)),
    'world-inventory-Stone': spaces.Box(*_unbound, shape=(1,)),
    'world-inventory-Wood': spaces.Box(*_unbound, shape=(1,)),
    'time': spaces.Box(*_unbound, shape=(1, 1)),

    'Build-build_payment': spaces.Box(*_unbound, shape=(1,)),
    'Build-build_skill': spaces.Box(*_unbound, shape=(1,)),

    'ContinuousDoubleAuction-market_rate-Stone': spaces.Box(*_unbound, shape=(1,)),
    'ContinuousDoubleAuction-price_history-Stone': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-available_asks-Stone': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-available_bids-Stone': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-my_asks-Stone': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-my_bids-Stone': spaces.Box(*_unbound, shape=(11,)),

    'ContinuousDoubleAuction-market_rate-Wood': spaces.Box(*_unbound, shape=(1,)),
    'ContinuousDoubleAuction-price_history-Wood': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-available_asks-Wood': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-available_bids-Wood': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-my_asks-Wood': spaces.Box(*_unbound, shape=(11,)),
    'ContinuousDoubleAuction-my_bids-Wood': spaces.Box(*_unbound, shape=(11,)),

    'Gather-bonus_gather_prob': spaces.Box(*_unbound, shape=(1,)),

    'action_mask': spaces.Box(*_unbound, shape=(50,)),

})
ACT_SPACE_AGENT = spaces.Discrete(50)


class AIEEnv(MultiAgentEnv):
    def __init__(self, env_config, force_dense_logging: bool = False):
        self.env: LayoutFromFile = foundation.make_env_instance(**{
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
