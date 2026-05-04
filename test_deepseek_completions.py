import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

payload = {
    "model": "deepseek-chat",
    "prompt": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 10
}
req = urllib.request.Request(
    "https://api.deepseek.com/beta/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
)
try:
    with urllib.request.urlopen(req) as resp:
        print("Success:", resp.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
