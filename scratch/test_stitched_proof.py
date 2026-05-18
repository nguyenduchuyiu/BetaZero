
import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.env.lean_env import Lean4ServerScheduler

def get_node_proof(node_id, nodes, out_edges):
    node = nodes[node_id]
    if node["type"] == "OR":
        # Find the solved AND child
        solved_child = None
        for tgt, rel in out_edges.get(node_id, []):
            if nodes[tgt]["status"] == "SOLVED":
                solved_child = tgt
                break
        if solved_child:
            return get_node_proof(solved_child, nodes, out_edges)
        else:
            return "sorry"
    else: # AND
        if node["action_type"] == "tactic":
            return node["extracted_lean_code"]
        elif node["action_type"] == "skeleton":
            skeleton_code = node["extracted_lean_code"]
            # Find child states
            child_states = []
            for tgt, rel in out_edges.get(node_id, []):
                if nodes[tgt]["type"] == "OR":
                    child_states.append(tgt)
            
            # Get proofs of children
            child_proofs = [get_node_proof(cid, nodes, out_edges) for cid in child_states]
            
            # Stitch them into the skeleton
            stitched = ProofStitcher.stitch(skeleton_code, child_proofs)
            return stitched

def test_stitched_proof():
    with open("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/aime_1983_p1.json", "r") as f:
        data = json.load(f)

    nodes = {n["id"]: n for n in data.get("nodes", [])}
    edges = data.get("edges", [])

    # Build relationships
    out_edges = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        rel = edge["relation"]
        out_edges.setdefault(src, []).append((tgt, rel))

    # Stitch the full proof starting from state_0
    full_body = get_node_proof("state_0", nodes, out_edges)
    
    # Wrap in the main theorem signature
    state_0 = nodes["state_0"]
    content = state_0["content"]
    if isinstance(content, str):
        content = json.loads(content)
        
    header = "import Mathlib\n\nopen BigOperators Nat Real Topology\n"
    full_proof_code = f"""{header}
theorem my_theorem (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 < w) (h0 : logb ↑x ↑w = 24) (h1 : logb ↑y ↑w = 40) (h2 : logb (↑x * ↑y * ↑z) ↑w = 12) : logb ↑z ↑w = 60 := by
{full_body}
"""

    print("=== FULL STITCHED PROOF ===")
    print(full_proof_code)
    
    # Save and verify
    Path("repl").mkdir(exist_ok=True)
    with open("repl/stitched_aime.lean", "w") as f:
        f.write(full_proof_code)
        
    scheduler = Lean4ServerScheduler(timeout=30)
    try:
        vr = scheduler.verify(full_proof_code)
        print("\n=== VERIFICATION RESULT ===")
        print("Pass:", vr.get("pass"))
        print("Complete:", vr.get("complete"))
        print("Errors:")
        for err in vr.get("errors", []):
            print(f"L{err.get('pos', {}).get('line')}:{err.get('pos', {}).get('column')} - {err.get('data')}")
    finally:
        scheduler.close()

if __name__ == "__main__":
    test_stitched_proof()
