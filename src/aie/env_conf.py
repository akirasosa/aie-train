import numpy as np

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
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'fixed-bracket-rates',
            'fixed_bracket_rates': np.zeros(7).tolist(),
        }),
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

ENV_CONF_COMMUNISM = {
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'fixed-bracket-rates',
            # Communism
            'fixed_bracket_rates': np.ones(7).tolist(),
        }),
    ],
}
ENV_CONF_MACHINE = {
    'scenario_name': 'PlannerLikeAgentsEnv',
    # only productivity is rewarded
    'mixing_weight_gini_vs_coin': 1,
}

ENV_CONF_DYSTOPIA = {
    'scenario_name': 'PlannerLikeAgentsEnv',
    # weights equality and productivity equally
    'mixing_weight_gini_vs_coin': 0,
}
