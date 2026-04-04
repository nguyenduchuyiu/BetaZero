import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# ==========================================
# PHẦN TEST SCRIPT MOCK ĐỂ IN CÂY
# ==========================================

from betazero.core import ProofState
from betazero.search import LevelwiseRollout
from betazero.env.lean_env import LeanEnv
from betazero.search import Sorrifier
from betazero.search import ANDORGraph
from betazero.search import RewardCalculator
from betazero.env import Lean4ServerScheduler


class MockPolicy:
    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]:
        batches = []
        for state in states:
            goal_text = state.goal.strip()
            
            # Khởi tạo mảng rỗng, sau đó nhét đủ n phần tử
            batch = []
            if "a + b = b + a" in goal_text: # Root state
                if action_type == "tactic":
                    batch = ["exact 100", "rw [Nat.add_comm]"]
                else:
                    batch = [
                        "have h_fail : a = b := by rfl\nsorry",   # Skel lỗi logic
                        "have h_clean : a = a := by sorry\nsorry" # Skel mượt
                    ]
            elif "a = a" in goal_text: # Subgoal sinh ra từ h_clean
                if action_type == "tactic":
                    batch = ["exact 99", "rfl"]
                else:
                    batch = ["sorry", "sorry"]
            else:
                # Các state rác khác (như a = b), cho tịt ngòi hết
                batch = ["sorry"] * n
                
            # Đảm bảo trả về đúng K action (nếu K > 2 thì bù thêm sorry)
            batch = (batch + ["sorry"] * n)[:n]
            batches.append(batch)
            
        return batches


def print_graph(graph: ANDORGraph, state: ProofState, depth: int = 0, visited: set = None):
    """Hàm in đồ thị đẹp như mơ để m soi các node đẻ ra có chuẩn không"""
    if visited is None: visited = set()
    indent = "   " * depth
    
    if state in visited:
        print(f"{indent}⭕ State: {state.goal.splitlines()[0]:<20} [Đã duyệt]")
        return
    visited.add(state)
    
    print(f"{indent}🔵 State: {state.goal.splitlines()[0]}")
    actions = graph.get_actions(state)
    
    if not actions:
        print(f"{indent}   (Không có action nào)")
        return
        
    for i, act in enumerate(actions):
        status = graph.status(act)
        reward = graph.get_r_env(act)
        content_preview = act.content.strip().replace('\n', '\\n')
        if len(content_preview) > 40: content_preview = content_preview[:37] + "..."
        
        branch_char = "└──" if i == len(actions) - 1 else "├──"
        print(f"{indent}   {branch_char} 🔲 [{act.action_type.upper()}] ({status}) [r_env: {reward:>4.1f}]: {content_preview}")
        
        for child in act.children:
            print_graph(graph, child, depth + 1, visited)

if __name__ == "__main__":
    # 1. Khởi tạo môi trường thật
    verifier = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="auto_sorrifier_cli")
    lean_env = LeanEnv(verifier)
    sorrifier = Sorrifier(verifier)
    
    # 2. Khởi tạo mock
    policy = MockPolicy()
    reward = RewardCalculator()
    
    # 3. Chạy LevelwiseRollout với K=4 (2 tac, 2 skel)
    rollout = LevelwiseRollout(
        policy=policy, lean=lean_env, sorrifier=sorrifier, reward=reward,
        K=4, max_depth=2, max_nodes=20
    )
    
    # Đề bài: Chứng minh tính giao hoán của phép cộng
    root_state = ProofState(context="a b : Nat", goal="a + b = b + a")
    
    print("\n🚀 BẮT ĐẦU CHẠY ROLLOUT...\n")
    _, graph, _ = rollout.rollout(root_state)
    
    print("🌳 ĐỒ THỊ AND/OR SAU KHI ROLLOUT:")
    print_graph(graph, root_state)