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
    report_content = []
    report_content.append("# Gemini 3 Flash miniF2F-test Rollout Performance & Node Stats (Full 244-Problem Set)\n\n")
    report_content.append(f"**Total Problems:** {total_problems}\n")
    report_content.append(f"**Total Solved:** {total_solved} / {total_problems} (**{total_solved/total_problems*100:.2f}%**)\n")
    report_content.append(f"- **Direct Solved at Root (Depth 0):** {direct_solved} / {total_problems} (**{direct_solved/total_problems*100:.2f}%**)\n")
    report_content.append(f"- **Hierarchical Solved (Skeleton Depth >= 1):** {hierarchical_solved} / {total_problems} (**{hierarchical_solved/total_problems*100:.2f}%**)\n\n")
    
    # Averages
    avg_nodes_solved = sum(r["total_nodes"] for r in results if r["solved"]) / max(1, total_solved)
    avg_nodes_failed = sum(r["total_nodes"] for r in results if not r["solved"]) / max(1, total_problems - total_solved)
    avg_tokens_solved = sum(r["est_tokens"] for r in results if r["solved"]) / max(1, total_solved)
    avg_tokens_failed = sum(r["est_tokens"] for r in results if not r["solved"]) / max(1, total_problems - total_solved)
    total_tokens_all = sum(r["est_tokens"] for r in results)
    
    report_content.append("## 1. Summary Averages\n\n")
    report_content.append("| Metrics | Solved Problems | Failed Problems | All Combined |\n")
    report_content.append("| :--- | :---: | :---: | :---: |\n")
    report_content.append(f"| **Average Total Nodes** | {avg_nodes_solved:.1f} | {avg_nodes_failed:.1f} | {sum(r['total_nodes'] for r in results)/total_problems:.1f} |\n")
    report_content.append(f"| **Average Tactic Nodes** | {sum(r['tactic_nodes'] for r in results if r['solved'])/max(1, total_solved):.1f} | {sum(r['tactic_nodes'] for r in results if not r['solved'])/max(1, total_problems - total_solved):.1f} | {sum(r['tactic_nodes'] for r in results)/total_problems:.1f} |\n")
    report_content.append(f"| **Average Skeleton Nodes** | {sum(r['skeleton_nodes'] for r in results if r['solved'])/max(1, total_solved):.1f} | {sum(r['skeleton_nodes'] for r in results if not r['solved'])/max(1, total_problems - total_solved):.1f} | {sum(r['skeleton_nodes'] for r in results)/total_problems:.1f} |\n")
    report_content.append(f"| **Average Est. Output Tokens** | {avg_tokens_solved:,.0f} | {avg_tokens_failed:,.0f} | {total_tokens_all/total_problems:,.0f} |\n")
    report_content.append(f"| **Total Output Tokens Generated** | - | - | **{total_tokens_all:,}** |\n\n")
    
    report_content.append("## 2. Direct Root Solves vs Hierarchical Solves Breakdown\n\n")
    report_content.append(f"### Direct Solves at Root (Depth 0) — {direct_solved} Problems\n")
    report_content.append("These problems were solved by directly applying tactic steps on the root goal, without requiring skeleton decomposition:\n\n")
    for i, r in enumerate([x for x in results if x["is_direct"]], 1):
        report_content.append(f"{i}. `{r['name']}` ({r['total_nodes']} nodes, {r['est_tokens']:,} output tokens)\n")
        
    report_content.append(f"\n### Hierarchical Solves (Depth >= 1) — {hierarchical_solved} Problems\n")
    report_content.append("These problems were solved using GammaZero's signature nested skeleton proof decomposition search:\n\n")
    for i, r in enumerate([x for x in results if x["solved"] and not x["is_direct"]], 1):
        report_content.append(f"{i}. `{r['name']}` ({r['total_nodes']} nodes, {r['skeleton_nodes']} skeletons, {r['est_tokens']:,} output tokens)\n")
        
    report_content.append("\n## 3. Detailed Problem Statistics\n\n")
    report_content.append("| No. | Problem Name | Status | Total Nodes | Tactic Nodes | Skeleton Nodes | Est. Output Tokens |\n")
    report_content.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for i, r in enumerate(results, 1):
        if r["solved"]:
            status_str = "**SOLVED (Direct)**" if r["is_direct"] else "**SOLVED (Hierarchical)**"
        else:
            status_str = "FAILED"
        report_content.append(f"| {i} | `{r['name']}` | {status_str} | {r['total_nodes']} | {r['tactic_nodes']} | {r['skeleton_nodes']} | {r['est_tokens']:,} |\n")

    full_text = "".join(report_content)
    
    # Save to both paths
    paths = [
        "/workspace/npthai/BetaZero/scratch/test_rollout_tokens_report.md",
        "/workspace/npthai/BetaZero/test_rollout_tokens_report.md",
        "/home/npthai/.gemini/antigravity/brain/5a1fc07e-dc22-4f49-bac2-0c1f69b4f581/test_rollout_tokens_report.md"
    ]
    for p in paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(full_text)

    print(f"Total Solved: {total_solved} / {total_problems} ({total_solved/total_problems*100:.2f}%)")
    print(f"Direct Solved (Depth 0): {direct_solved} / {total_problems} ({direct_solved/total_problems*100:.2f}%)")
    print(f"Hierarchical Solved (Depth >= 1): {hierarchical_solved} / {total_problems} ({hierarchical_solved/total_problems*100:.2f}%)")

if __name__ == "__main__":
    analyze()
