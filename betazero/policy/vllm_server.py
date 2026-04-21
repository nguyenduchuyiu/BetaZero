import os
import signal
import subprocess
import time
import requests
import socket

from betazero.core import ProofState
from betazero.policy.prompt import build_prompt
from betazero.utils import Config


class VLLMServer:
    """vLLM server as a subprocess. kill() → OS reclaims 100% VRAM."""

    def __init__(self, cfg: Config):
        self.model_name       = cfg.model_name
        self.base_port        = cfg.vllm_port
        self.port             = self.base_port
        self.gpu_util         = cfg.vllm_gpu_memory_utilization
        self.max_tokens       = cfg.max_new_tokens
        self.temperature      = cfg.temperature
        self.max_model_len    = cfg.max_model_len
        self.ready_timeout    = cfg.vllm_ready_timeout
        self.proc: subprocess.Popen | None = None
        self.log_file = None
        self._adapter_flag: bool | None = None  # Cache for whether adapter is loaded

    def _get_free_port(self, start_port: int) -> int:
        """Scan from start_port, find the first empty port and return it."""
        port = start_port
        while port < start_port + 1000: # Scan up to 1000 ports
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    # Try to bind to the port. If OS allows -> Port is empty!
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    # Error (Already in use) -> OS rejects -> Try next port
                    port += 1
        raise RuntimeError(f"Scan 1000 ports from {start_port} but no empty port found!")

    def start(self, adapter_path: str | None = None):
        """Spawn vLLM subprocess; block until /health is up."""
        self.active_adapter_path = adapter_path
        
        self.port = self._get_free_port(self.base_port)
        env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_util),
            "--max-model-len", str(self.max_model_len),
            "--enable-lora"
        ]
        
        self._loaded_loras = set()
        lora_modules = []
        if adapter_path and os.path.exists(adapter_path):
            lora_modules.append(f"adapter={adapter_path}")
        
        if os.path.exists("qwen_lora_tactic"):
            lora_modules.append("tactic=qwen_lora_tactic")
            self._loaded_loras.add("tactic")
        if os.path.exists("qwen_lora_skeleton"):
            lora_modules.append("skeleton=qwen_lora_skeleton")
            self._loaded_loras.add("skeleton")

        if lora_modules:
            cmd += ["--enable-lora", "--max-loras", str(max(2, len(lora_modules))), "--lora-modules"] + lora_modules
        else:
            cmd += ["--enable-lora", "--max-loras", "2"]
        print(f"\n[vLLM] Starting on port {self.port}...")

        os.makedirs("outputs", exist_ok=True)
        self.log_file = open(f"outputs/vllm_port_{self.port}.log", "w")

        self.proc = subprocess.Popen(
            cmd, 
            env=env,
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            preexec_fn=os.setsid     
        )
        self._adapter_flag = None  # Reset adapter cached state at startup
        self._wait_ready(self.ready_timeout)

    def kill(self):
        """Kill subprocess AND ALL ITS CHILDREN; VRAM is fully reclaimed by the OS."""
        if self.proc:
            try:
                if self.proc.poll() is None:
                    print(f"[vLLM] Killing Process Group {self.proc.pid}...")
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.proc.wait()
            finally:
                self.proc = None

        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        
        self._adapter_flag = None  # Clear adapter cached state at shutdown
        
        time.sleep(2)

    def sample(
        self,
        states: list[ProofState],
        action_type: str,
        n: int,
        *,
        prompts: list[str] | None = None,
    ) -> list[list[dict]]:
        """Sample `n` completions per state row. Returns len(states) lists, each of length `n`."""
        if not states or n <= 0:
            return [[] for _ in states]
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]
        elif len(prompts) != len(states):
            raise ValueError("prompts length must match states")
        
        lora_name = f"{action_type}_active" if (hasattr(self, 'active_adapter_path') and self.active_adapter_path) else action_type
        model = self.model_name

        if hasattr(self, "_loaded_loras") and lora_name in self._loaded_loras:
            model = lora_name
        else:
            lora_path = ""
            if hasattr(self, "active_adapter_path") and self.active_adapter_path:
                active_path = os.path.join(self.active_adapter_path, action_type)
                if os.path.exists(active_path):
                    lora_path = active_path
            
            if not lora_path:
                fallback = getattr(self.cfg, f"base_lora_{action_type}", f"qwen_lora_{action_type}")
                lora_path = os.path.abspath(fallback)
                
            if os.path.exists(lora_path):
                try:
                    res = requests.post(f"http://localhost:{self.port}/v1/load_lora_adapter", json={
                        "lora_name": lora_name,
                        "lora_path": lora_path,
                        "load_inplace": True
                    }, timeout=10)
                    if res.ok:
                        if not hasattr(self, "_loaded_loras"): 
                            self._loaded_loras = set()
                        self._loaded_loras.add(lora_name)
                        model = lora_name
                    else:
                        print(f"[vLLM] failed dynamically loading {lora_name}: {res.text}")
                except requests.exceptions.RequestException as e:
                    print(f"[vLLM] load_lora_adapter error: {e}")
            else:
                if self._adapter_flag is None:
                    self._adapter_flag = self._adapter_loaded()
                if self._adapter_flag:
                    model = "adapter"

        try:
            r = requests.post(
                f"http://localhost:{self.port}/v1/completions",
                json={
                      "model": model, 
                      "prompt": prompts, 
                      "n": n,
                      "max_tokens": self.max_tokens, 
                      "temperature": self.temperature,
                      "stop": ["<|end|>", "<|endoftext|>", "<|im_end|>"]
                    },
                timeout=(10, 1800),
            )
            r.raise_for_status()
            choices = r.json().get("choices", [])
        except requests.exceptions.RequestException as e:
            print(f"[vLLM] sample error: {e}")
            return [[] for _ in states]
        
        def get_choice(c):
            return {"text": c["text"].strip()}

        return [[get_choice(choices[i * n + j]) for j in range(n)]
                for i in range(len(states))]

    def _adapter_loaded(self) -> bool:
        try:
            r = requests.get(f"http://localhost:{self.port}/v1/models", timeout=2)
            return any(m["id"] == "adapter" for m in r.json().get("data", []))
        except Exception:
            return False

    def _wait_ready(self, timeout: int):
        for _ in range(timeout):
            try:
                if requests.get(f"http://localhost:{self.port}/health", timeout=1).ok:
                    return
            except Exception:
                pass
            time.sleep(1)
        self.kill()
        raise RuntimeError(f"vLLM did not start within {timeout}s")
