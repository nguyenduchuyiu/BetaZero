import json
import numpy as np
import sys

def analyze(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    r_envs = []
    r_deps = []
    q_values = []
    
    statuses = {}
    node_types = {}
    
    for node in nodes:
        status = node.get('status', 'UNKNOWN')
        statuses[status] = statuses.get(status, 0) + 1
        
        node_type = node.get('type', 'UNKNOWN')
        node_types[node_type] = node_types.get(node_type, 0) + 1

        metrics = node.get('metrics', {})
        if 'r_env' in metrics:
            r_envs.append(metrics['r_env'])
        if 'r_dep' in metrics:
            r_deps.append(metrics['r_dep'])
        if 'Q_value' in metrics:
            q_values.append(metrics['Q_value'])
            
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
        "metrics": [
            get_stats(r_envs, "r_env"),
            get_stats(r_deps, "r_dep"),
            get_stats(q_values, "Q_value")
        ],
        "statuses": statuses,
        "node_types": node_types,
        "total_nodes": len(nodes)
    }
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    analyze(sys.argv[1])
