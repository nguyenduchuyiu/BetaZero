import os
import json
import glob

def analyze():
    directory = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    results = []
    solved_count = 0
    total_problems = len(json_files)
    
    for path in json_files:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {name}: {e}")
                continue
                
        root_id = data.get("root_id", "state_0")
        nodes = data.get("nodes", [])
        
        # Check status
        node_status = {}
        for n in nodes:
            node_status[n["id"]] = n["status"]
            
        is_solved = (node_status.get(root_id) == "SOLVED")
        if is_solved:
            solved_count += 1
            
        total_nodes = data.get("total_nodes", len(nodes))
        
        budget = data.get("search_metadata", {}).get("budget", {})
        used_total = budget.get("used_total", 0)
        used_tactic = budget.get("used_tactic", 0)
        used_skeleton = budget.get("used_skeleton_raw", 0)
        lean_calls = budget.get("lean_verify_calls", 0)
        
        results.append({
            "name": name,
            "solved": "SOLVED" if is_solved else "FAILED",
            "is_solved_bool": is_solved,
            "total_nodes": total_nodes,
            "used_total_actions": used_total,
            "used_tactic": used_tactic,
            "used_skeleton": used_skeleton,
            "lean_calls": lean_calls
        })
        
    # Sort results: solved first, then by name
    results.sort(key=lambda x: (0 if x["is_solved_bool"] else 1, x["name"]))
    
    # Generate Markdown report
    md = []
    md.append("# miniF2F-test Individual Problem Rollout Metrics\n")
    md.append(f"**Total Problems:** {total_problems}  ")
    md.append(f"**Solved:** {solved_count} / {total_problems}  ")
    md.append(f"**Solve Rate:** {solved_count / max(1, total_problems) * 100:.1f}%\n")
    
    md.append("| No. | Theorem Name | Status | Total Nodes | Total Actions | Tactic Actions | Skeleton Actions | Lean Calls |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for idx, r in enumerate(results, 1):
        status_str = f"**{r['solved']}**" if r['is_solved_bool'] else f"*{r['solved']}*"
        md.append(f"| {idx} | `{r['name']}` | {status_str} | {r['total_nodes']} | {r['used_total_actions']} | {r['used_tactic']} | {r['used_skeleton']} | {r['lean_calls']} |")
        
    report_content = "\n".join(md)
    
    # Save inside workspace
    workspace_path = "/workspace/npthai/BetaZero/test_split_individual_problem_metrics.md"
    with open(workspace_path, "w") as out:
        out.write(report_content)
        
    # Save inside artifacts
    artifact_path = "/home/npthai/.gemini/antigravity/brain/5a1fc07e-dc22-4f49-bac2-0c1f69b4f581/test_split_individual_problem_metrics.md"
    with open(artifact_path, "w") as out:
        out.write(report_content)
        
    print(f"Analyzed {total_problems} files. Solved: {solved_count} ({solved_count/total_problems*100:.1f}%).")
    print("Reports written successfully.")

if __name__ == "__main__":
    analyze()
