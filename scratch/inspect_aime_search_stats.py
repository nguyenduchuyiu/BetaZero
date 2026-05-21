import json

def inspect_details():
    path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/aime_1991_p9.json"
    with open(path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    node_map = {n["id"]: n for n in nodes}
    
    # Let's count how many children each OR node has
    or_children = {}
    for edge in edges:
        if edge["relation"] == "expanded_to":
            src = edge["source"]
            tgt = edge["target"]
            or_children.setdefault(src, []).append(tgt)
            
    print("OR Node children count & status:")
    for node in nodes:
        if node["type"] == "OR":
            children = or_children.get(node["id"], [])
            print(f"  OR Node {node['id']} (Depth {node['depth']}): status={node['status']}, children_count={len(children)}, goal={node.get('content', {}).get('goal', '')[:50].strip()}")

if __name__ == "__main__":
    inspect_details()
