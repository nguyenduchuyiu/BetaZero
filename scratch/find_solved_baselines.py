import os
import json

baseline_dir = "outputs/baseline_logs"
solved_files = []

for filename in os.listdir(baseline_dir):
    if filename.endswith(".json"):
        path = os.path.join(baseline_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            if summary.get("solved") == True or summary.get("passed_count", 0) > 0:
                solved_files.append((filename, summary))
        except Exception as e:
            pass

print(f"Total solved baseline problems: {len(solved_files)}")
for f, s in solved_files[:10]:
    print(f"  {f}: {s}")
