import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

from gammazero.core import ProofState
from gammazero.policy.prompt import build_prompt
from gammazero.utils import Config

from dotenv import load_dotenv
load_dotenv()

class GeminiAPIServer:
    """API server policy for parallel generation using Google Gemini API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview") 
        self.batch_size = getattr(cfg, "api_batch_size", 16)
        self.max_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.max_retries = 3
        self.retry_delay = 2.0

        if not self.api_key:
            print("Warning: GOOGLE_API_KEY environment variable is not set!")
        
        self.client = genai.Client(api_key=self.api_key)

    def start(self, adapter_path: str | None = None):
        """No background process to start for API, just placeholder."""
        print(f"\n[Gemini API] Initialized parallel API mode (batch_size={self.batch_size}, model={self.model}).")

    def kill(self):
        """No process to kill."""
        pass

    def _get_gemini_response(self, prompt: str) -> dict | None:
        try:
            # Disable thinking as requested: 'không thinking'            
            if os.getenv("DEBUG_GEMINI"):
                print(f"DEBUG: Calling Gemini with max_output_tokens={self.max_tokens}")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
                
            )
            
            finish_reason = ""
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason or "")
                if finish_reason != "STOP":
                    print(f"Warning: Gemini stopped with reason: {response.candidates[0].finish_reason}")
                    if response.candidates[0].finish_message:
                        print(f"Finish message: {response.candidates[0].finish_message}")

            text = ""
            if response.text:
                text = response.text
            elif response.candidates and response.candidates[0].content.parts:
                text = "".join(p.text for p in response.candidates[0].content.parts if p.text)
            return {"text": text, "finish_reason": finish_reason}
        except Exception as e:
            print(f"Lỗi gọi Gemini API: {e}")
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
            prompts = [build_prompt(s, action_type) for s in states]
        elif len(prompts) != len(states):
            raise ValueError("prompts length must match states")

        results = [[] for _ in states]
        
        requests_args = []
        for i, prompt in enumerate(prompts):
            for _ in range(n):
                requests_args.append((i, prompt))
                
        def _one_request(req_idx: int, args):
            i, prompt = args
            started = time.time()
            for attempt in range(1, self.max_retries + 1):
                res = self._get_gemini_response(prompt)
                if res is not None:
                    elapsed = time.time() - started
                    print(
                        f"[Gemini API] Received completion {req_idx + 1}/{len(requests_args)} "
                        f"(state={i}, attempt={attempt}, chars={len(res.get('text', ''))}, elapsed={elapsed:.1f}s)",
                        flush=True,
                    )
                    return req_idx, i, res
                time.sleep(self.retry_delay)
            elapsed = time.time() - started
            print(
                f"[Gemini API] Completion {req_idx + 1}/{len(requests_args)} failed after "
                f"{self.max_retries} attempts (state={i}, elapsed={elapsed:.1f}s)",
                flush=True,
            )
            return req_idx, i, {"text": ""}

        print(f"[Gemini API] Generating {len(requests_args)} completions in parallel (batch_size={self.batch_size})...")
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(_one_request, req_idx, args)
                for req_idx, args in enumerate(requests_args)
            ]
            outcomes = [future.result() for future in as_completed(futures)]
            
        for _, i, res in sorted(outcomes, key=lambda item: item[0]):
            results[i].append(res)
                
        return results
