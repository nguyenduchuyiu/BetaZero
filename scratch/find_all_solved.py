import os
import json

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
                
                # Case 1: Search tree format (contains "nodes")
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
                                chars = len(pb)
                                solved_proofs[filename] = {
                                    "lines": lines,
                                    "chars": chars,
                                    "format": "tree"
                                }
                
                # Case 2: Sampling/rollout baseline format (contains "samples" and "summary")
                elif "samples" in data and "summary" in data:
                    summary = data.get("summary", {})
                    # Checked if solved
                    if summary.get("solved") == True or summary.get("passed_count", 0) > 0:
                        # Find the first sample that has verification.pass == True
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
                                # We can extract the actual proof body (lines after ":= by" or ":= ")
                                # But counting the total lines of the Lean code file is also very consistent.
                                # Let's count both!
                                total_lines = len(code.strip().split("\n"))
                                
                                # Attempt to extract proof body lines
                                proof_body = ""
                                if ":= by" in code:
                                    proof_body = code.split(":= by", 1)[1]
                                elif ":=" in code:
                                    proof_body = code.split(":=", 1)[1]
                                
                                if proof_body:
                                    pb_lines = len(proof_body.strip().split("\n"))
                                else:
                                    pb_lines = total_lines
                                    
                                solved_proofs[filename] = {
                                    "lines": pb_lines,
                                    "chars": len(code),
                                    "format": "sampling"
                                }
            except Exception as e:
                pass
    return solved_proofs

print("Analyzing baseline_logs...")
baseline = analyze_directory("outputs/baseline_logs")
print(f"Baseline solved: {len(baseline)}")

print("\nAnalyzing rollouts/miniF2F-valid...")
valid = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-valid")
print(f"Valid solved: {len(valid)}")

print("\nAnalyzing rollouts/miniF2F-test...")
test = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-test")
print(f"Test solved: {len(test)}")

# Let's see intersection between baseline and either valid/test
valid_and_test = {**valid, **test}
print(f"\nTotal Scaffold-Aware Solved: {len(valid_and_test)}")

intersection = set(baseline.keys()) & set(valid_and_test.keys())
print(f"Intersection (solved in both): {len(intersection)}")

print("\nMatched Solved Problems:")
for f in sorted(list(intersection)):
    b_len = baseline[f]["lines"]
    s_len = valid_and_test[f]["lines"]
    print(f"  {f}: Non-Scaffold={b_len} lines, Scaffold-Aware={s_len} lines")
