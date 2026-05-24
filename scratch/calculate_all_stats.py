import os
import json

baseline_dir = "outputs/baseline_logs"
scaffold_dir = "outputs/rollouts/gemini3flash/miniF2F-valid"

def get_solved_proof_lengths(directory):
    solved_proofs = {}
    if not os.path.exists(directory):
        return solved_proofs
        
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                # Check if state_0 is solved
                nodes = data.get("nodes", [])
                # If nodes is a list
                if isinstance(nodes, list):
                    state_0 = next((n for n in nodes if n.get("id") == "state_0"), None)
                elif isinstance(nodes, dict):
                    state_0 = nodes.get("state_0")
                else:
                    state_0 = None
                    
                if state_0 and state_0.get("status") == "SOLVED":
                    content = state_0.get("content")
                    if isinstance(content, dict) and "proof_body" in content:
                        pb = content["proof_body"]
                        if pb:
                            lines = len(pb.strip().split("\n"))
                            chars = len(pb)
                            solved_proofs[filename] = {
                                "lines": lines,
                                "chars": chars
                            }
            except Exception as e:
                pass
    return solved_proofs

baseline_solved = get_solved_proof_lengths(baseline_dir)
scaffold_solved = get_solved_proof_lengths(scaffold_dir)

print(f"Baseline (Non-Scaffold) solved: {len(baseline_solved)}")
print(f"Scaffold-Aware solved: {len(scaffold_solved)}")

# Find intersection
intersection = set(baseline_solved.keys()) & set(scaffold_solved.keys())
print(f"Intersection (solved in both): {len(intersection)}")

# Let's print out the list of matching files and their lengths
print("\nMatching solved files:")
for f in sorted(list(intersection))[:15]:
    b_len = baseline_solved[f]["lines"]
    s_len = scaffold_solved[f]["lines"]
    print(f"  {f}: Non-Scaffold={b_len} lines, Scaffold-Aware={s_len} lines")
