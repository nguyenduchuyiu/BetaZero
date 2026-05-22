import json

path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/algebra_apbmpcneq0_aeq0anbeq0anceq0.json"
with open(path, "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
for node in nodes[:5]:
    print("Node ID:", node.get("id"))
    print("Node Type:", node.get("type"))
    print("Node Keys:", list(node.keys()))
    if "goal" in node:
        print("Node Goal:", node["goal"][:100])
    if "state" in node:
        print("Node State:", str(node["state"])[:100])
    print("-" * 40)
