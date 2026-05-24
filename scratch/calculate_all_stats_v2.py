import os
import json
import numpy as np

def analyze_directory(directory):
    solved_proofs = {}
    if not os.path.exists(directory):
        return solved_proofs
        
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                if "nodes" in data:
                    nodes = data["nodes"]
                    state_0 = None
                    if isinstance(nodes, list):
                        state_0 = next((n for n in nodes if n.get("id") == "state_0"), None)
                    elif isinstance(nodes, dict):
                        state_0 = nodes.get("state_0")
                    
                    if state_0 and state_0.get("status") == "SOLVED":
                        content = state_0.get("content")
                        if isinstance(content, dict) and "proof_body" in content:
                            pb = content["proof_body"]
                            if pb:
                                lines = len(pb.strip().split("\n"))
                                solved_proofs[filename] = lines
                
                elif "samples" in data and "summary" in data:
                    summary = data.get("summary", {})
                    if summary.get("solved") == True or summary.get("passed_count", 0) > 0:
                        samples = data.get("samples", [])
                        passed_sample = None
                        for s in samples:
                            ver = s.get("verification")
                            if isinstance(ver, dict) and ver.get("pass") == True:
                                passed_sample = s
                                break
                        
                        if passed_sample:
                            code = passed_sample.get("extracted_code")
                            if code:
                                total_lines = len(code.strip().split("\n"))
                                proof_body = ""
                                if ":= by" in code:
                                    proof_body = code.split(":= by", 1)[1]
                                elif ":=" in code:
                                    proof_body = code.split(":=", 1)[1]
                                
                                if proof_body:
                                    pb_lines = len(proof_body.strip().split("\n"))
                                else:
                                    pb_lines = total_lines
                                solved_proofs[filename] = pb_lines
            except Exception as e:
                pass
    return solved_proofs

baseline = analyze_directory("outputs/baseline_logs")
valid = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-valid")
test = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-test")
scaffold = {**valid, **test}

baseline_all_vals = list(baseline.values())
scaffold_all_vals = list(scaffold.values())

intersection = sorted(list(set(baseline.keys()) & set(scaffold.keys())))
ns_inter_vals = [baseline[f] for f in intersection]
s_inter_vals = [scaffold[f] for f in intersection]

print(f"=== STATISTICS ON ALL SOLVED PROBLEMS ===")
print(f"Non-Scaffold-Aware Baseline (N={len(baseline_all_vals)}):")
print(f"  Mean: {np.mean(baseline_all_vals):.2f} lines")
print(f"  Median: {np.median(baseline_all_vals):.2f} lines")
print(f"  Max: {np.max(baseline_all_vals)} lines")
print(f"  Min: {np.min(baseline_all_vals)} lines")

print(f"\nScaffold-Aware MCTS (N={len(scaffold_all_vals)}):")
print(f"  Mean: {np.mean(scaffold_all_vals):.2f} lines")
print(f"  Median: {np.median(scaffold_all_vals):.2f} lines")
print(f"  Max: {np.max(scaffold_all_vals)} lines")
print(f"  Min: {np.min(scaffold_all_vals)} lines")

print(f"\nGlobal Average Reduction (All Solved): {((np.mean(baseline_all_vals) - np.mean(scaffold_all_vals)) / np.mean(baseline_all_vals)) * 100:.2f}%")
print(f"Global Median Reduction (All Solved): {((np.median(baseline_all_vals) - np.median(scaffold_all_vals)) / np.median(baseline_all_vals)) * 100:.2f}%")

print(f"\n=== STATISTICS ON INTERSECTION ONLY (N={len(intersection)}) ===")
print(f"Non-Scaffold-Aware Baseline:")
print(f"  Mean: {np.mean(ns_inter_vals):.2f} lines")
print(f"  Median: {np.median(ns_inter_vals):.2f} lines")
print(f"Scaffold-Aware MCTS:")
print(f"  Mean: {np.mean(s_inter_vals):.2f} lines")
print(f"  Median: {np.median(s_inter_vals):.2f} lines")
