from ai_economist.foundation.base.base_env import scenario_registry

from aie.environments.planner_like_agents_env import PlannerLikeAgentsEnv

scenario_registry.add(PlannerLikeAgentsEnv)
