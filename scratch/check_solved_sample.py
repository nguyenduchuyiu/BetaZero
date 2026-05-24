import json

path = "outputs/baseline_logs/algebra_2rootsintpoly_am10tap11eqasqpam110.json"
with open(path, "r") as f:
    data = json.load(f)

samples = data.get("samples", [])
print(f"Total samples: {len(samples)}")
# Let's find one that succeeded
passed_samples = [s for s in samples if s.get("status") == "PASSED" or (isinstance(s.get("verification"), dict) and s["verification"].get("success") == True)]
print(f"Passed samples: {len(passed_samples)}")
if passed_samples:
    first_passed = passed_samples[0]
    print("Keys in passed sample:", list(first_passed.keys()))
    print("Status:", first_passed.get("status"))
    print("Verification success:", first_passed.get("verification"))
    print("extracted_code sample (first 200 chars):")
    print(first_passed.get("extracted_code")[:200])
