import os
import json
import glob

def analyze_rollouts(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Found {len(json_files)} rollout JSON files to analyze.")

    total_theorems = len(json_files)
    solved_theorems = 0
    total_actions = 0
    total_lean_calls = 0
    solved_depths = []
    total_pruned_global = 0
    total_pruned_per_depth = 0
    
    total_repaired_candidates = 0
    total_tactic_attempts = 0
    total_skeleton_attempts = 0
    
    total_committed = 0
    total_reserved = 0
    total_fallback = 0
    
    r_env_scores = []
    r_dep_scores = []
    
    theorem_results = []

    for path in sorted(json_files):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        name = os.path.basename(path).replace(".json", "")
        
        # 1. Proof Success
        root_solved = False
        root_id = data.get("root_id", "state_0")
        
        # Find root status
        for node in data.get("nodes", []):
            if node.get("id") == root_id:
                root_solved = (node.get("status") == "SOLVED")
                break
        
        if root_solved:
            solved_theorems += 1
            
        # Find max depth of solved OR nodes
        max_solved_depth = 0
        for node in data.get("nodes", []):
            if node.get("type") == "OR" and node.get("status") == "SOLVED":
                max_solved_depth = max(max_solved_depth, node.get("depth", 0))
        if root_solved:
            solved_depths.append(max_solved_depth)

        # 2. Search Cost & Metadata
        metadata = data.get("search_metadata", {})
        budget = metadata.get("budget", {})
        
        verify_calls = budget.get("lean_verify_calls", 0) + budget.get("patch_verify_calls", 0)
        total_lean_calls += verify_calls
        
        beam = metadata.get("beam", {})
        total_pruned_global += beam.get("states_pruned_global", 0)
        total_pruned_per_depth += beam.get("states_pruned_per_depth", 0)

        # 3. Actions
        file_actions = 0
        file_repaired = 0
        file_tactics = 0
        file_skeletons = 0
        
        for node in data.get("nodes", []):
            if node.get("type") == "AND":
                file_actions += 1
                action_type = node.get("action_type", "")
                if action_type == "tactic":
                    file_tactics += 1
                elif action_type == "skeleton":
                    file_skeletons += 1
                
                # Check if it was repaired
                if node.get("patched_code") and node.get("patched_code").strip():
                    file_repaired += 1
                elif "[SYNTHETIC_PATCH]" in node.get("prompt", ""):
                    file_repaired += 1
                
                # Collect scores
                metrics = node.get("metrics", {})
                if "r_env" in metrics:
                    r_env_scores.append(metrics["r_env"])
                if "r_dep" in metrics:
                    r_dep_scores.append(metrics["r_dep"])
        
        total_actions += file_actions
        total_repaired_candidates += file_repaired
        total_tactic_attempts += file_tactics
        total_skeleton_attempts += file_skeletons

        # 4. Decomposition
        commitment = metadata.get("skeleton_commitment", {})
        total_committed += commitment.get("committed", 0)
        total_reserved += commitment.get("reserved", 0)
        total_fallback += commitment.get("fallback_activated", 0)
        
        theorem_results.append({
            "name": name,
            "solved": root_solved,
            "actions": file_actions,
            "lean_calls": verify_calls,
            "max_solved_depth": max_solved_depth if root_solved else "-"
        })

    solve_rate = (solved_theorems / total_theorems) * 100 if total_theorems > 0 else 0
    avg_actions = total_actions / total_theorems if total_theorems > 0 else 0
    avg_lean_calls = total_lean_calls / total_theorems if total_theorems > 0 else 0
    avg_solved_depth = sum(solved_depths) / len(solved_depths) if solved_depths else 0
    avg_pruned = (total_pruned_global + total_pruned_per_depth) / total_theorems if total_theorems > 0 else 0
    
    repair_rate = (total_repaired_candidates / total_actions) * 100 if total_actions > 0 else 0
    avg_r_env = sum(r_env_scores) / len(r_env_scores) if r_env_scores else 0
    avg_r_dep = sum(r_dep_scores) / len(r_dep_scores) if r_dep_scores else 0

    print("\n================== METRICS ANALYSIS ==================")
    print(f"Total Theorems Analyzed: {total_theorems}")
    print(f"Solved: {solved_theorems} / {total_theorems} ({solve_rate:.2f}%)")
    print(f"Total Actions: {total_actions} (Tactics: {total_tactic_attempts}, Skeletons: {total_skeleton_attempts})")
    print(f"Total Lean Verification Calls: {total_lean_calls}")
    print(f"Average Actions per Theorem: {avg_actions:.2f}")
    print(f"Average Lean Calls per Theorem: {avg_lean_calls:.2f}")
    print(f"Average Solved Depth: {avg_solved_depth:.2f}")
    print(f"Average States Pruned per Theorem: {avg_pruned:.2f}")
    print(f"Total Repaired Candidates: {total_repaired_candidates} ({repair_rate:.2f}%)")
    print(f"Average r_env (Syntactic Survival): {avg_r_env:.4f}")
    print(f"Average r_dep (Dependency Reward): {avg_r_dep:.4f}")
    print(f"Skeleton Commitments: Committed={total_committed}, Reserved={total_reserved}, Fallback={total_fallback}")
    print("======================================================")

    # Output Markdown Table
    print("\n### Markdown Table for Paper:")
    print("| Setting | Solved | Solve Rate | Avg. Actions | Avg. Lean Calls |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    print(f"| **GammaZero (Test)** | {solved_theorems} / {total_theorems} | {solve_rate:.1f}% | {avg_actions:.1f} | {avg_lean_calls:.1f} |")
    
    print("\n### Detailed Per-Theorem Results:")
    print("| Theorem | Solved? | Generated Actions | Lean Calls | Max Solved Depth |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for tr in theorem_results:
        status = "✅ YES" if tr["solved"] else "❌ NO"
        print(f"| {tr['name']} | {status} | {tr['actions']} | {tr['lean_calls']} | {tr['max_solved_depth']} |")

if __name__ == "__main__":
    analyze_rollouts("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test")
