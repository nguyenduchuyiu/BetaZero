import json
import os
from betazero.core.nodes import ProofState, Action
from betazero.search.graph.and_or_graph import ANDORGraph

def generate_standard_log(filename: str, case_type: str):
    # Khởi tạo root
    root = ProofState(
        context="x : ℝ\nh₀ : 0 < x ∧ x < Real.pi",
        goal="12 ≤ (9 * (x ^ 2 * sin x ^ 2) + 4) / (x * sin x)"
    )
    g = ANDORGraph(root)
    
    # Action 0: Luôn có một hành động thất bại để giống log thật
    act0 = Action("tactic", "Initial guess: nlinarith", "nlinarith", children=())
    g.expand(root, act0, tactic_status="FAILED")
    
    # Action 1: Skeleton
    skel_code = """
  have h_core : a = b := by sorry
  have h_junk : 1 = 1 := by sorry
  exact h_core
"""
    sg_core = ProofState("...", "a = b")
    sg_junk = ProofState("...", "1 = 1")
    
    act_skel = Action("skeleton", "Sorrification", skel_code, (sg_core, sg_junk))
    g.expand(root, act_skel, r_env=0.1)

    if case_type == "clean":
        g.expand(sg_core, Action("tactic", "Solved core", "exact h_used"), tactic_status="SOLVED")
        g.expand(sg_junk, Action("tactic", "Solved junk", "rfl"), tactic_status="SOLVED")
        g.set_skeleton_override(act_skel, True)
        g.set_r_dep(act_skel, 1.0)
    elif case_type == "mixed":
        g.expand(sg_core, Action("tactic", "Solved core", "exact h_used"), tactic_status="SOLVED")
        # junk remains OPEN/FAILED
        g.set_skeleton_override(act_skel, True)
        g.set_r_dep(act_skel, 0.4)
    elif case_type == "failed":
        # core remains OPEN/FAILED
        g.expand(sg_junk, Action("tactic", "Solved junk", "rfl"), tactic_status="SOLVED")
        g.set_skeleton_override(act_skel, False)
        g.set_r_dep(act_skel, 0.0)

    q_values = g.backup()
    
    nodes = []
    edges = []
    state_to_id = {s: f"state_{i}" for i, s in enumerate(g._actions.keys())}
    action_to_id = {a: f"action_{i}" for i, a in enumerate(g._parent.keys())}

    for state, id_s in state_to_id.items():
        # Trích xuất proof body hoàn chỉnh nếu đã giải xong
        proof_body = g.extract_proof_code(state) if g.status(state) == "SOLVED" else "  sorry"
        
        nodes.append({
            "id": id_s, "type": "OR", "status": g.status(state), "depth": g.get_depth(state),
            "content": {
                "context": state.context, 
                "goal": state.goal,
                "proof_body": proof_body
            },
            "metrics": {"V_value": max((q_values.get(a, 0.0) for a in g.get_actions(state)), default=0.0)}
        })
        for act in g.get_actions(state):
            id_a = action_to_id[act]
            nodes.append({
                "id": id_a, "type": "AND", "action_type": act.action_type, "status": g.status(act),
                "content": act.content, "extracted_code": act.extracted_code,
                "metrics": {"r_env": g.get_r_env(act), "r_dep": g._r_dep.get(act, 0.0), "Q_value": q_values.get(act, 0.0)}
            })
            edges.append({"source": id_s, "target": id_a, "relation": "expanded_to"})
            for child in act.children:
                edges.append({"source": id_a, "target": state_to_id[child], "relation": "subgoal"})

    output = {"theorem_goal": root.goal, "root_id": "state_0", "total_nodes": len(nodes), "nodes": nodes, "edges": edges}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    generate_standard_log("tests/log_clean.json", "clean")
    generate_standard_log("tests/log_mixed.json", "mixed")
    generate_standard_log("tests/log_failed.json", "failed")
    print("✅ Đã xuất 3 file log kèm proof_body: log_clean.json, log_mixed.json, log_failed.json")
