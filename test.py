import requests
import json

url = "http://127.0.0.1:5000/v1/completions"

payload = {
    "model": "Qwen/Qwen2.5-0.5B",
    "prompt": "Hi!",
    "temperature": 1.0,
    "max_tokens": 512,
    "logprobs": True
}

resp = requests.post(url, json=payload)
resp.raise_for_status()

data = resp.json()

# ======================
# 1. SAVE RAW RESPONSE
# ======================
with open("test.txt", "w") as f:
    json.dump(data, f, indent=2)

print("Saved to test.txt")

# ======================
# 2. PRINT TEXT OUTPUT
# ======================
text = data["choices"][0]["text"]
print("\n=== OUTPUT ===")
print(text)

# ======================
# 3. LOGPROBS (GRPO USE)
# ======================
logprobs = data["choices"][0].get("logprobs", None)

if logprobs:
    print("\n=== TOKEN LOGPROBS ===")

    tokens = logprobs.get("tokens", [])
    token_logprobs = logprobs.get("token_logprobs", [])

    for t, lp in zip(tokens, token_logprobs):
        print(f"{t:>10} | {lp}")
