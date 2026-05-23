import os
import json

def classify_solved_problems():
    json_dir = "/workspace/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    
    groups = {
        "under_32": [],
        "under_64": [],
        "under_128": [],
        "under_256": [],
        "under_512": [],
        "others": []
    }
    
    total_benchmark_problems = 244
    
    for f in json_files:
        json_path = os.path.join(json_dir, f)
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            
        root_id = data.get("root_id", "state_0")
        root_node = next((n for n in data.get("nodes", []) if n.get("id") == root_id), None)
        
        # Only process solved problems
        if root_node and root_node.get("status") == "SOLVED":
            prob_name = f.replace(".json", "")
            
            # Extract nodes used
            used_nodes = 0
            metadata = data.get("search_metadata")
            if metadata and "budget" in metadata:
                used_nodes = metadata["budget"].get("used_total", 0)
            else:
                # Fallback: Count total OR (state) nodes in graph
                used_nodes = sum(1 for n in data.get("nodes", []) if n.get("type") == "OR")
                
            if used_nodes < 32:
                groups["under_32"].append((prob_name, used_nodes))
            elif used_nodes < 64:
                groups["under_64"].append((prob_name, used_nodes))
            elif used_nodes < 128:
                groups["under_128"].append((prob_name, used_nodes))
            elif used_nodes < 256:
                groups["under_256"].append((prob_name, used_nodes))
            elif used_nodes <= 512:
                groups["under_512"].append((prob_name, used_nodes))
            else:
                groups["others"].append((prob_name, used_nodes))
                
    # Print the results in a beautiful Markdown table
    total_solved = sum(len(g) for g in groups.values())
    unsolved_count = total_benchmark_problems - total_solved
    
    print("# Classifying Solved Problems by Search Budget (Nodes Used)\n")
    print(f"Total Benchmark Problems: {total_benchmark_problems}")
    print(f"Total Solved Problems: {total_solved} (Pass Rate: {total_solved/total_benchmark_problems:.1%})")
    print(f"Total Unsolved Problems: {unsolved_count} (Fail Rate: {unsolved_count/total_benchmark_problems:.1%})\n")
    
    print("| Category | Range (Nodes) | Count | Pass Rate Contribution | Cumulative Pass Rate |")
    print("|---|---|---|---|---|")
    
    cum_count = 0
    categories_info = [
        ("Under 32", "under_32", "`[0, 32)`"),
        ("Under 64", "under_64", "`[32, 64)`"),
        ("Under 128", "under_128", "`[64, 128)`"),
        ("Under 256", "under_256", "`[128, 256)`"),
        ("Under 512", "under_512", "`[256, 512]`")
    ]
    
    for label, key, node_range in categories_info:
        count = len(groups[key])
        cum_count += count
        contrib = count / total_benchmark_problems
        cum_rate = cum_count / total_benchmark_problems
        print(f"| **{label}** | {node_range} | {count} | {contrib:.1%} | {cum_rate:.1%} |")
        
    if groups["others"]:
        count = len(groups["others"])
        cum_count += count
        contrib = count / total_benchmark_problems
        cum_rate = cum_count / total_benchmark_problems
        print(f"| **More than 512** | `> 512` | {count} | {contrib:.1%} | {cum_rate:.1%} |")
        
    print(f"| **Unsolved** | - | {unsolved_count} | {unsolved_count/total_benchmark_problems:.1%} | 100.0% |")
    print("\n---\n")

if __name__ == "__main__":
    classify_solved_problems()
