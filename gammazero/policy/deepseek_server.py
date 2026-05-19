import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor

from gammazero.core import ProofState
from gammazero.policy.prompt import build_prompt
from gammazero.utils import Config

from dotenv import load_dotenv
load_dotenv()

class DeepSeekAPIServer:
    """API server policy for parallel generation using official DeepSeek API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/beta"
        self.model = "deepseek-v4-flash"
        self.batch_size = 16  # Parallel requests
        self.max_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.max_retries = 3
        self.retry_delay = 2.0

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def start(self, adapter_path: str | None = None):
        """No background process to start for API, just placeholder."""
        print(f"\n[API] Initialized parallel API mode (batch_size={self.batch_size}).")

    def kill(self):
        """No process to kill."""
        pass

    def _get_official_api_response(self, prompt: str) -> dict | None:
        if not self.api_key:
            print("Warning: DEEPSEEK_API_KEY environment variable is not set!")
            
        payload = {
            "model": self.model,
            "extra_body": {"thinking": {"type": "disabled"}},
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["<|end|>", "<|endoftext|>", "<|im_end|>"]
        }

        try:
            resp = self.session.post(
                f"{self.base_url}/completions",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return {
                "text": choice.get("text", ""),
                "finish_reason": choice.get("finish_reason", ""),
            }
        except Exception as e:
            print(f"Lỗi gọi Official API: {e}")
            return None

    def sample(
        self,
        states: list[ProofState],
        action_type: str,
        n: int,
        *,
        prompts: list[str] | None = None,
    ) -> list[list[dict]]:
        """Sample `n` completions per state using ThreadPoolExecutor."""
        if not states or n <= 0:
            return [[] for _ in states]
        if prompts is None:
            from gammazero.policy.prompt import build_prompt
            prompts = [build_prompt(s, action_type) for s in states]
        elif len(prompts) != len(states):
            raise ValueError("prompts length must match states")

        results = [[] for _ in states]
        
        requests_args = []
        for i, prompt in enumerate(prompts):
            for _ in range(n):
                requests_args.append((i, prompt))
                
        def _one_request(args):
            i, prompt = args
            for attempt in range(1, self.max_retries + 1):
                res = self._get_official_api_response(prompt)
                if res is not None:
                    return i, res
                time.sleep(self.retry_delay)
            return i, {"text": ""}

        print(f"[API] Generating {len(requests_args)} completions in parallel (batch_size={self.batch_size})...")
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            outcomes = list(executor.map(_one_request, requests_args))
            
        for i, res in outcomes:
            if res["text"]:
                results[i].append(res)
            else:
                results[i].append({"text": ""})
                
        return results
