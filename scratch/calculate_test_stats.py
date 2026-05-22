import os
import json
import glob

def analyze():
    test_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    json_files = glob.glob(os.path.join(test_dir, "*.json"))
    json_files.sort()
    
    results = []
    total_solved = 0
    direct_solved = 0      # Solved at root node (0 skeletons in search graph)
    hierarchical_solved = 0  # Solved with skeleton search (>=1 skeletons in search graph)
    total_problems = len(json_files)
    
    for path in json_files:
        name = os.path.basename(path).replace(".json", "")
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
                
        root_id = data.get("root_id", "state_0")
        nodes = data.get("nodes", [])
        
        # Check solve status
        root_solved = False
        for n in nodes:
            if n["id"] == root_id:
                root_solved = (n["status"] == "SOLVED")
                break
        
        # Count node types
        total_nodes = data.get("total_nodes", len(nodes))
        tactic_nodes = 0
        skeleton_nodes = 0
        or_nodes = 0
        
        llm_chars = 0
        for n in nodes:
            ntype = n.get("type")
            if ntype == "OR":
                or_nodes += 1
            elif ntype == "AND":
                act_type = n.get("action_type", "")
                if act_type == "tactic":
                    tactic_nodes += 1
                elif act_type == "skeleton":
                    skeleton_nodes += 1
                
                # Sum generated texts
                lean_code = n.get("extracted_lean_code", "")
                if isinstance(lean_code, str):
                    llm_chars += len(lean_code)
                
                patched = n.get("patched_code", "")
                if isinstance(patched, str):
                    llm_chars += len(patched)
                elif isinstance(patched, dict):
                    for val in patched.values():
                        if isinstance(val, str):
                            llm_chars += len(val)
                            
        est_tokens = int(llm_chars / 4.0)
        
        is_direct = False
        if root_solved:
            total_solved += 1
            if skeleton_nodes == 0:
                direct_solved += 1
                is_direct = True
            else:
                hierarchical_solved += 1
        
        results.append({
            "name": name,
            "solved": root_solved,
            "is_direct": is_direct,
            "total_nodes": total_nodes,
            "tactic_nodes": tactic_nodes,
            "skeleton_nodes": skeleton_nodes,
            "or_nodes": or_nodes,
            "est_tokens": est_tokens
        })
        
    # Write full detailed report to markdown
    report_path = "/workspace/npthai/BetaZero/scratch/test_rollout_tokens_report.md"
    with open(report_path, "w") as f:
        f.write("# Gemini 3 Flash miniF2F-test Rollout Performance & Node Stats\n\n")
        f.write(f"**Total Problems:** {total_problems}\n")
        f.write(f"**Total Solved:** {total_solved} / {total_problems} (**{total_solved/total_problems*100:.2f}%**)\n")
        f.write(f"- **Direct Solved at Root (Depth 0):** {direct_solved} / {total_problems} (**{direct_solved/total_problems*100:.2f}%**)\n")
        f.write(f"- **Hierarchical Solved (Skeleton Depth >= 1):** {hierarchical_solved} / {total_problems} (**{hierarchical_solved/total_problems*100:.2f}%**)\n\n")
        
        # Averages
        avg_nodes_solved = sum(r["total_nodes"] for r in results if r["solved"]) / max(1, total_solved)
        avg_nodes_failed = sum(r["total_nodes"] for r in results if not r["solved"]) / max(1, total_problems - total_solved)
        avg_tokens_solved = sum(r["est_tokens"] for r in results if r["solved"]) / max(1, total_solved)
        avg_tokens_failed = sum(r["est_tokens"] for r in results if not r["solved"]) / max(1, total_problems - total_solved)
        total_tokens_all = sum(r["est_tokens"] for r in results)
        
        f.write("## 1. Summary Averages\n\n")
        f.write("| Metrics | Solved Problems | Failed Problems | All Combined |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Average Total Nodes** | {avg_nodes_solved:.1f} | {avg_nodes_failed:.1f} | {sum(r['total_nodes'] for r in results)/total_problems:.1f} |\n")
        f.write(f"| **Average Tactic Nodes** | {sum(r['tactic_nodes'] for r in results if r['solved'])/max(1, total_solved):.1f} | {sum(r['tactic_nodes'] for r in results if not r['solved'])/max(1, total_problems - total_solved):.1f} | {sum(r['tactic_nodes'] for r in results)/total_problems:.1f} |\n")
        f.write(f"| **Average Skeleton Nodes** | {sum(r['skeleton_nodes'] for r in results if r['solved'])/max(1, total_solved):.1f} | {sum(r['skeleton_nodes'] for r in results if not r['solved'])/max(1, total_problems - total_solved):.1f} | {sum(r['skeleton_nodes'] for r in results)/total_problems:.1f} |\n")
        f.write(f"| **Average Est. Output Tokens** | {avg_tokens_solved:,.0f} | {avg_tokens_failed:,.0f} | {total_tokens_all/total_problems:,.0f} |\n")
        f.write(f"| **Total Output Tokens Generated** | - | - | **{total_tokens_all:,}** |\n\n")
        
        f.write("## 2. Direct Root Solves vs Hierarchical Solves Breakdown\n\n")
        f.write(f"### Direct Solves at Root (Depth 0) — {direct_solved} Problems\n")
        f.write("These problems were solved by directly applying tactic steps on the root goal, without requiring skeleton decomposition:\n\n")
        for i, r in enumerate([x for x in results if x["is_direct"]], 1):
            f.write(f"{i}. `{r['name']}` ({r['total_nodes']} nodes, {r['est_tokens']:,} output tokens)\n")
            
        f.write(f"\n### Hierarchical Solves (Depth >= 1) — {hierarchical_solved} Problems\n")
        f.write("These problems were solved using GammaZero's signature nested skeleton proof decomposition search:\n\n")
        for i, r in enumerate([x for x in results if x["solved"] and not x["is_direct"]], 1):
            f.write(f"{i}. `{r['name']}` ({r['total_nodes']} nodes, {r['skeleton_nodes']} skeletons, {r['est_tokens']:,} output tokens)\n")
            
        f.write("\n## 3. Detailed Problem Statistics\n\n")
        f.write("| No. | Problem Name | Status | Total Nodes | Tactic Nodes | Skeleton Nodes | Est. Output Tokens |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for i, r in enumerate(results, 1):
            if r["solved"]:
                status_str = "**SOLVED (Direct)**" if r["is_direct"] else "**SOLVED (Hierarchical)**"
            else:
                status_str = "FAILED"
            f.write(f"| {i} | `{r['name']}` | {status_str} | {r['total_nodes']} | {r['tactic_nodes']} | {r['skeleton_nodes']} | {r['est_tokens']:,} |\n")

    print(f"Total Solved: {total_solved} / {total_problems} ({total_solved/total_problems*100:.2f}%)")
    print(f"Direct Solved (Depth 0): {direct_solved} / {total_problems} ({direct_solved/total_problems*100:.2f}%)")
    print(f"Hierarchical Solved (Depth >= 1): {hierarchical_solved} / {total_problems} ({hierarchical_solved/total_problems*100:.2f}%)")

if __name__ == "__main__":
    analyze()
