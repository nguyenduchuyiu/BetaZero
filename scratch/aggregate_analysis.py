import json
import os
import numpy as np
from collections import Counter

def aggregate_stats(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    total_theorems = len(files)
    solved_theorems = 0
    total_nodes_all = 0
    
    r_envs = []
    r_deps = []
    q_values = []
    
    global_statuses = Counter()
    global_node_types = Counter()
    
    theorem_results = []

    # Track Synthetic vs Normal skeletons
    synthetic_stats = {"count": 0, "r_envs": [], "r_deps": [], "solved": 0}
    normal_stats = {"count": 0, "r_envs": [], "r_deps": [], "solved": 0}

    for filename in files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        nodes = data.get('nodes', [])
        total_nodes_all += len(nodes)
        
        # Check root status
        root_node = next((n for n in nodes if n.get('id') == 'state_0'), None)
        is_solved = root_node and root_node.get('status') == 'SOLVED'
        if is_solved: solved_theorems += 1
            
        file_r_envs = []
        for node in nodes:
            global_statuses[node.get('status', 'UNKNOWN')] += 1
            global_node_types[node.get('type', 'UNKNOWN')] += 1
            
            metrics = node.get('metrics', {})
            r_e = metrics.get('r_env')
            r_d = metrics.get('r_dep')
            
            if r_e is not None:
                r_envs.append(r_e)
                file_r_envs.append(r_e)
            if r_d is not None:
                r_deps.append(r_d)
            if 'Q_value' in metrics:
                q_values.append(metrics['Q_value'])

            # Phân tích sâu nhóm Skeleton (AND nodes)
            if node.get('type') == "AND" and node.get('action_type') == "skeleton":
                prompt = node.get('prompt', '')
                is_synth = prompt.startswith("[SYNTHETIC_PATCH]")
                target = synthetic_stats if is_synth else normal_stats
                
                target["count"] += 1
                if r_e is not None: target["r_envs"].append(r_e)
                if r_d is not None: target["r_deps"].append(r_d)
                if node.get('status') == "SOLVED": target["solved"] += 1
        
        avg_r_env = np.mean(file_r_envs) if file_r_envs else 0
        theorem_results.append({
            "name": filename,
            "solved": is_solved,
            "nodes": len(nodes),
            "avg_r_env": float(avg_r_env)
        })

    def get_stats(vals, name):
        if not vals:
            return {"name": name, "msg": "No data"}
        vals = np.array(vals)
        return {
            "name": name,
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "zeros": int(np.sum(vals == 0)),
            "ones": int(np.sum(vals == 1.0))
        }

    def format_cat(stats, name):
        if stats["count"] == 0: return { "name": name, "msg": "N/A" }
        return {
            "name": name,
            "count": stats["count"],
            "avg_r_env": float(np.mean(stats["r_envs"])) if stats["r_envs"] else 0,
            "avg_r_dep": float(np.mean(stats["r_deps"])) if stats["r_deps"] else 0,
            "solved_count": stats["solved"],
            "solve_rate": stats["solved"] / stats["count"]
        }

    results = {
        "summary": {
            "total_theorems": total_theorems,
            "solved_theorems": solved_theorems,
            "pass_rate": solved_theorems / total_theorems if total_theorems > 0 else 0,
            "total_nodes": total_nodes_all,
            "avg_nodes_per_theorem": total_nodes_all / total_theorems if total_theorems > 0 else 0
        },
        "skeleton_analysis": [
            format_cat(normal_stats, "Normal Skeletons"),
            format_cat(synthetic_stats, "Synthetic Patches")
        ],
        "metrics": [
            get_stats(r_envs, "r_env"),
            get_stats(r_deps, "r_dep"),
            get_stats(q_values, "Q_value")
        ],
        "statuses": dict(global_statuses),
        "node_types": dict(global_node_types)
    }
    
    # Sort theorems by r_env to see best/worst
    theorem_results.sort(key=lambda x: x['avg_r_env'], reverse=True)
    results["top_theorems"] = theorem_results[:10]
    results["bottom_theorems"] = theorem_results[-10:]
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import sys
    aggregate_stats(sys.argv[1])
