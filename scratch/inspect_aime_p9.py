import json

def inspect_aime():
    path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/aime_1991_p9.json"
    with open(path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_map = {n["id"]: n for n in nodes}
    
    # We want to identify the AND nodes that were actually used in the final proof.
    # The final proof is reconstructed by starting from state_0, and recursively finding the solved AND node child,
    # and for that AND node, all its children OR nodes (subgoals).
    
    children_map = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        children_map.setdefault(src, []).append(tgt)
        
    used_and_nodes = []
    used_or_nodes = []
    
    def traverse(node_id):
        node = node_map.get(node_id)
        if not node:
            return
        if node.get("type") == "OR":
            used_or_nodes.append(node_id)
            # Find the solved children AND node(s)
            children = children_map.get(node_id, [])
            for c_id in children:
                child = node_map.get(c_id)
                if child and child.get("type") == "AND" and child.get("status") == "SOLVED":
                    # In an OR node, we only need ONE solved child
                    # Let's traverse it and then break (since one solved path is enough for proof)
                    traverse(c_id)
                    break
        elif node.get("type") == "AND":
            used_and_nodes.append(node_id)
            # Traverse all children OR nodes (subgoals)
            children = children_map.get(node_id, [])
            for c_id in children:
                traverse(c_id)
                
    traverse("state_0")
    
    print(f"Total used AND nodes: {len(used_and_nodes)}")
    for a_id in used_and_nodes:
        node = node_map[a_id]
        metrics = node.get("metrics", {})
        code = node.get("extracted_lean_code", "").strip().split("\n")[0]
        print(f"AND Node {a_id}: r_env={metrics.get('r_env')}, r_dep={metrics.get('r_dep')}, Q={metrics.get('Q_value')}, code={code[:50]}")

if __name__ == "__main__":
    inspect_aime()
