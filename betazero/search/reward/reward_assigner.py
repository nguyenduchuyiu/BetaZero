from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph
from .calculator import RewardCalculator


class DependencyRewardAssigner:
    """Set r_dep on skeleton actions from Lean dependency structure."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def assign(self, graph: ANDORGraph) -> None:
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue
            lean_code = get_lean_code(action.content)
            state_code = self.lean._build_cmd(parent_state, lean_code)
            graph.set_r_dep(action, self.reward.r_dep(self.lean.dep_graph(state_code)))
