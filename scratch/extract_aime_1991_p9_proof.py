import json
import os

path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/aime_1991_p9.json"
with open(path, "r") as f:
    data = json.load(f)

print("=== Theorem Goal ===")
print(data.get("theorem_goal"))
print("Total Nodes:", data.get("total_nodes"))

nodes = data.get("nodes", [])
edges = data.get("edges", [])

node_status = {}
node_type = {}
node_label = {}
node_content = {}
node_depth = {}

for n in nodes:
    nid = n["id"]
    node_status[nid] = n["status"]
    node_type[nid] = n["type"]
    node_label[nid] = n.get("target_label", "")
    node_depth[nid] = n.get("depth", 0)
    
    # Action nodes keys:
    if n["type"] == "AND":
        node_content[nid] = n.get("stitched_code", n.get("content", ""))

expanded_to = {}
subgoals = {}

for edge in edges:
    src = edge["source"]
    tgt = edge["target"]
    rel = edge["relation"]
    if rel == "expanded_to":
        expanded_to.setdefault(src, []).append(tgt)
    elif rel == "subgoal":
        subgoals.setdefault(src, []).append(tgt)

root_id = data.get("root_id", "state_0")

# Find the chosen solved action at the root
root_actions = expanded_to.get(root_id, [])
solved_root_actions = [act for act in root_actions if node_status.get(act) == "SOLVED"]

print(f"\nRoot solved actions count: {len(solved_root_actions)}")
if solved_root_actions:
    root_act = solved_root_actions[0]
    print(f"Root Solved Action ID: {root_act}")
    print("=== ROOT STITCHED LEAN 4 PROOF ===")
    
    # Let's find the stitched code or proof of this action
    # Let's inspect the keys of this action node
    for n in nodes:
        if n["id"] == root_act:
            print("Action Type:", n.get("action_type"))
            print("Keys:", list(n.keys()))
            print("\nPROOF CONTENT:\n")
            print(n.get("stitched_code") or n.get("patched_code") or n.get("content"))
            
            # Let's save it to a separate file for the user
            proof_code = n.get("stitched_code") or n.get("patched_code") or n.get("content")
            if isinstance(proof_code, dict):
                proof_code = proof_code.get("proof_body", "")
            
            with open("/workspace/npthai/BetaZero/scratch/aime_1991_p9_proof.lean", "w") as out:
                out.write(str(proof_code))
            print("\nSaved proof to /workspace/npthai/BetaZero/scratch/aime_1991_p9_proof.lean")

# Let's trace the solved tree
print("\n=== Solved Tree Traversal ===")
def print_tree(state_id, depth=0):
    status = node_status.get(state_id)
    label = node_label.get(state_id, "")
    goal = ""
    for n in nodes:
        if n["id"] == state_id:
            content = n.get("content", {})
            if isinstance(content, dict):
                goal = content.get("goal", "")
            break
            
    print("  " * depth + f"- State {state_id} (Label: {label}, Status: {status}, Depth: {node_depth.get(state_id)}, Goal: {goal})")
    
    # Get solved action
    acts = expanded_to.get(state_id, [])
    solved_acts = [act for act in acts if node_status.get(act) == "SOLVED"]
    if solved_acts:
        act = solved_acts[0]
        # Get action details
        act_type = ""
        for n in nodes:
            if n["id"] == act:
                act_type = n.get("action_type", "")
                break
        print("  " * depth + f"  * Action {act} (Type: {act_type})")
        for child in subgoals.get(act, []):
            print_tree(child, depth + 2)

print_tree(root_id)
