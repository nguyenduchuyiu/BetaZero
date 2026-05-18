
import json

with open("/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/aime_1983_p1.json", "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
or_nodes = [n for n in nodes if n.get("type") == "OR"]
print(f"Total OR nodes: {len(or_nodes)}")

empty_ctx_count = 0
for node in or_nodes:
    content = node.get("content", {})
    ctx = content.get("context", "")
    goal = content.get("goal", "")
    if not ctx:
        empty_ctx_count += 1
        print(f"Node {node['id']} has empty context! Goal: {repr(goal)}")

print(f"OR nodes with empty context: {empty_ctx_count}")
