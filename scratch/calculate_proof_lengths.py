import json

def get_proof_lengths(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    lengths = []
    
    for n in nodes:
        if n.get("type") == "OR" and n.get("status") == "SOLVED":
            content = n.get("content")
            if isinstance(content, dict) and "proof_body" in content:
                pb = content["proof_body"]
                if pb:
                    lines = len(pb.strip().split("\n"))
                    chars = len(pb)
                    lengths.append({
                        "id": n.get("id"),
                        "lines": lines,
                        "chars": chars
                    })
    return lengths

non_scaffold_lengths = get_proof_lengths("aime_1983_p1-non-scaffold.json")
scaffold_lengths = get_proof_lengths("outputs/rollouts/gemini3flash/miniF2F-valid/aime_1983_p1.json")

print("=== Non-Scaffold-Aware ===")
print(f"Total solved OR nodes: {len(non_scaffold_lengths)}")
# Print list of sorted lengths
ns_sorted = sorted([x["lines"] for x in non_scaffold_lengths])
print("Lines distribution:", ns_sorted)
root_ns = next((x for x in non_scaffold_lengths if x["id"] == "state_0"), None)
print(f"Root proof length (state_0): {root_ns}")

print("\n=== Scaffold-Aware ===")
print(f"Total solved OR nodes: {len(scaffold_lengths)}")
s_sorted = sorted([x["lines"] for x in scaffold_lengths])
print("Lines distribution:", s_sorted)
root_s = next((x for x in scaffold_lengths if x["id"] == "state_0"), None)
print(f"Root proof length (state_0): {root_s}")
