import json

path = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-valid/aime_1991_p9.json"
with open(path, "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
for n in nodes:
    if n["id"] == "state_0":
        print("=== Root State Scaffold Code ===")
        print(n.get("scaffold_code"))
        print("\n=== Root State Content ===")
        print(n.get("content"))
        break
