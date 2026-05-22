import json

path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/algebra_apbmpcneq0_aeq0anbeq0anceq0.json"
with open(path, "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
for node in nodes:
    if node.get("type") == "OR":
        print("Node ID:", node.get("id"))
        print("Node Target Label:", node.get("target_label"))
        print("Node Content (first 200 chars):", str(node.get("content"))[:200])
        print("=" * 60)
