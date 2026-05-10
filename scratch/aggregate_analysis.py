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
        
        # Check root status (usually the first node or id state_0)
        root_node = next((n for n in nodes if n.get('id') == 'state_0'), None)
        if root_node and root_node.get('status') == 'SOLVED':
            solved_theorems += 1
            is_solved = True
        else:
            is_solved = False
            
        file_r_envs = []
        for node in nodes:
            global_statuses[node.get('status', 'UNKNOWN')] += 1
            global_node_types[node.get('type', 'UNKNOWN')] += 1
            
            metrics = node.get('metrics', {})
            if 'r_env' in metrics:
                r_envs.append(metrics['r_env'])
                file_r_envs.append(metrics['r_env'])
            if 'r_dep' in metrics:
                r_deps.append(metrics['r_dep'])
            if 'Q_value' in metrics:
                q_values.append(metrics['Q_value'])
        
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

    results = {
        "summary": {
            "total_theorems": total_theorems,
            "solved_theorems": solved_theorems,
            "pass_rate": solved_theorems / total_theorems if total_theorems > 0 else 0,
            "total_nodes": total_nodes_all,
            "avg_nodes_per_theorem": total_nodes_all / total_theorems if total_theorems > 0 else 0
        },
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
