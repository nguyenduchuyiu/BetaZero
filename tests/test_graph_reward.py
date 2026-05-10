import json
import os
from betazero.core.nodes import ProofState, Action
from betazero.search.graph.and_or_graph import ANDORGraph
from betazero.search.reward.calculator import RewardCalculator
from betazero.search.reward.reward_assigner import DependencyRewardAssigner
from betazero.env.lean_env import LeanEnv

def export_graph_to_json(graph: ANDORGraph, q_values: dict, filename: str):
    """Xuất đồ thị ra định dạng JSON phân cấp để dễ dàng visualize."""
    nodes = []
    links = []
    
    # Duyệt qua các state (OR nodes)
    for state in graph._actions.keys():
        state_id = f"state_{id(state)}"
        nodes.append({
            "id": state_id,
            "type": "state",
            "goal": state.goal,
            "status": graph.status(state),
            "depth": graph.get_depth(state)
        })
        
        # Duyệt qua các action (AND nodes) của state này
        for action in graph.get_actions(state):
            action_id = f"action_{id(action)}"
            nodes.append({
                "id": action_id,
                "type": "action",
                "action_type": action.action_type,
                "code": action.extracted_code,
                "status": graph.status(action),
                "r_env": graph.get_r_env(action),
                "r_dep": graph._r_dep.get(action, 0.0),
                "Q_value": q_values.get(action, 0.0)
            })
            
            # Link từ state -> action
            links.append({"source": state_id, "target": action_id})
            
            # Link từ action -> các state con
            for child in action.children:
                child_id = f"state_{id(child)}"
                links.append({"source": action_id, "target": child_id})
                
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2, ensure_ascii=False)

def run_test_cases():
    lean_env = LeanEnv(scheduler=None)
    reward_calc = RewardCalculator(W_solve=1.0, gamma=0.99)
    assigner = DependencyRewardAssigner(lean=lean_env, reward=reward_calc)

    # CASE 1: Mixed (Core Solved, Benign Solved, Malignant Open)
    print("\n--- TEST CASE 1: Mixed ---")
    root1 = ProofState(context="a b : Nat\nh : a = b", goal="a = b")
    g1 = ANDORGraph(root1)
    skel1 = Action("skeleton", "Mixed Skel", 
                  "  have h_core : a = b := by \n    sorry\n  have h_m : 2 = 2 := by \n    sorry\n  exact h_core", 
                  (ProofState("...", "a = b"), ProofState("...", "2 = 2")))
    g1.expand(root1, skel1, r_env=0.1)
    g1.expand(skel1.children[0], Action("tactic", "exact h", "    exact h"), r_env=1.0, tactic_status="SOLVED")
    assigner.assign(g1)
    q1 = g1.backup()
    export_graph_to_json(g1, q1, "tests/graph_case1.json")

    # CASE 2: Core Failed (r_dep should be 0.0)
    print("\n--- TEST CASE 2: Core Failed ---")
    root2 = ProofState(context="a b : Nat\nh : a = b", goal="a = b")
    g2 = ANDORGraph(root2)
    skel2 = Action("skeleton", "Failed Skel", 
                  "  have h_core : a = b := by \n    sorry\n  exact h_core", 
                  (ProofState("...", "a = b"),))
    g2.expand(root2, skel2, r_env=0.1)
    # Không giải h_core -> core_failed
    assigner.assign(g2)
    q2 = g2.backup()
    export_graph_to_json(g2, q2, "tests/graph_case2.json")

    # CASE 3: Pure Clean Proof (r_dep should be 1.0)
    print("\n--- TEST CASE 3: Clean ---")
    root3 = ProofState(context="a b : Nat\nh : a = b", goal="a = b")
    g3 = ANDORGraph(root3)
    skel3 = Action("skeleton", "Clean Skel", 
                  "  have h_core : a = b := by \n    sorry\n  exact h_core", 
                  (ProofState("...", "a = b"),))
    g3.expand(root3, skel3, r_env=0.1)
    g3.expand(skel3.children[0], Action("tactic", "exact h", "    exact h"), r_env=1.0, tactic_status="SOLVED")
    assigner.assign(g3)
    q3 = g3.backup()
    export_graph_to_json(g3, q3, "tests/graph_case3.json")

    print("\n✅ Đã xuất 3 file JSON test cases.")

if __name__ == "__main__":
    run_test_cases()
