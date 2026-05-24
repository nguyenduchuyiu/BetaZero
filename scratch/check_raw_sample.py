import json

path = "outputs/baseline_logs/algebra_2rootsintpoly_am10tap11eqasqpam110.json"
with open(path, "r") as f:
    data = json.load(f)

samples = data.get("samples", [])
print(f"Total samples: {len(samples)}")
if samples:
    first = samples[0]
    print("Keys in first sample:")
    for k, v in first.items():
        if isinstance(v, str):
            print(f"  {k}: string of len {len(v)}, sample: {v[:100]}...")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
            # Print verification contents
            if k == "verification":
                print(f"    verification content: {v}")
        else:
            print(f"  {k}: {v} (type: {type(v)})")
