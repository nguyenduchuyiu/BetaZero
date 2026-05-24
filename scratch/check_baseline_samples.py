import json

path = "outputs/baseline_logs/aimeII_2020_p6.json"
with open(path, "r") as f:
    data = json.load(f)

print("summary:", data.get("summary"))
samples = data.get("samples", [])
print(f"Number of samples: {len(samples)}")
if samples:
    first_sample = samples[0]
    print("Keys in first sample:", list(first_sample.keys()))
    print("Sample success:", first_sample.get("success") or first_sample.get("is_solved") or first_sample.get("solved"))
    print("Sample text sample (first 200 chars):")
    print(str(first_sample.get("text") or first_sample.get("output") or first_sample.get("proof"))[:200])
