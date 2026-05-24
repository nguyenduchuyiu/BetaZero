import json
import os

path = "outputs/baseline_logs/aimeII_2020_p6.json"
with open(path, "r") as f:
    data = json.load(f)

print("Keys:", list(data.keys()))
nodes = data.get("nodes", [])
print("Type of nodes:", type(nodes))
if isinstance(nodes, list) and len(nodes) > 0:
    print("First node keys:", list(nodes[0].keys()))
    print("First node id:", nodes[0].get("id"))
    print("First node status:", nodes[0].get("status"))
    print("First node content keys:", list(nodes[0].get("content", {}).keys()) if isinstance(nodes[0].get("content"), dict) else type(nodes[0].get("content")))
