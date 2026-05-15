
from betazero.search.graph import ANDORGraph
from betazero.core import ProofState, Action
from betazero.search.reward import RewardCalculator
from betazero.search.rollout.batch_executor import BatchExecutor, RolloutBudget
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
import os

def test():
    theorem = ProofState(context="n : ℕ", goal="n + 0 = n")
    graph = ANDORGraph(theorem)
    
    # Giả lập verifier
    class MockScheduler:
        def verify(self, code):
            return {"complete": True, "pass": True, "sorries": [], "messages": []}
    
    class MockLeanEnv:
        def __init__(self):
            self.scheduler = MockScheduler()
            self.scheduler.executor = None
        def execute(self, state, code):
            return "code", {"complete": True, "pass": True}, []
        def verify(self, code):
            return {"complete": True, "pass": True}

        def analyze_dependencies(self, code, allowed_vars=None):
            return {"core_solved": ["h1"], "benign": [], "malignant": []}

    reward = RewardCalculator()
    executor = BatchExecutor(MockLeanEnv(), None, reward)
    budget = RolloutBudget(10)
    
    # Giả lập 1 tactic solved
    state = theorem
    raw_output = "```lean4\ntheorem my_theorem (n : ℕ) : n + 0 = n := by\n  rfl\n```"
    action_batches = [[{"text": raw_output}]]
    
    # Chúng ta cần Mock cái policy hoặc chạy execute trực tiếp
    # Nhưng execute() dùng ThreadPool và tasks.
    # Thôi ta gọi expand trực tiếp như BatchExecutor làm xem r_env thế nào.
    
    r_env = 1.0 # Line 121
    act = Action(action_type="tactic", content=raw_output, extracted_code="rfl")
    graph.expand(state, act, r_env=r_env, tactic_status="SOLVED")
    
    print(f"r_env in graph: {graph.get_r_env(act)}")
    
    q_values = reward.compute_returns(graph)
    print(f"Q_value in graph: {q_values.get(act)}")

if __name__ == "__main__":
    test()
