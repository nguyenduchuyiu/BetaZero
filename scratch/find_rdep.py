import json
import os
import glob

def find_rdep_nodes(directory):
    files = glob.glob(os.path.join(directory, "*.json"))
    found_nodes = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            nodes = data.get('nodes', [])
            for node in nodes:
                metrics = node.get('metrics', {})
                r_dep = metrics.get('r_dep')
                
                if r_dep is not None:
                    # Check if r_dep is not 0 and not 1
                    # Using a small epsilon for float comparison if necessary, 
                    # but usually these are exactly 0.0 or 1.0 if they are those values.
                    if r_dep != 0 and r_dep != 1 and r_dep != 0.0 and r_dep != 1.0:
                        found_nodes.append({
                            'file': file_path,
                            'node_id': node.get('id'),
                            'r_dep': r_dep,
                            'extracted_lean_code': node.get('extracted_lean_code', '')[:100] # Snippet
                        })
                        if len(found_nodes) >= 10:
                            return found_nodes
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return found_nodes

if __name__ == "__main__":
    search_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid-50"
    results = find_rdep_nodes(search_dir)
    
    if not results:
        print("No nodes found with r_dep != 0 and r_dep != 1")
    for res in results:
        print(f"File: {res['file']}")
        print(f"  Node ID: {res['node_id']}")
        print(f"  r_dep: {res['r_dep']}")
        print(f"  Code: {res['extracted_lean_code']}...")
        print("-" * 40)
