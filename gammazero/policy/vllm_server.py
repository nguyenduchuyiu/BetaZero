import os
import signal
import subprocess
import time
import requests
import socket

from gammazero.core import ProofState
from gammazero.policy.prompt import build_prompt
from gammazero.utils import Config


class VLLMServer:
    """vLLM run as a subprocess. Calling kill() lets the OS reclaim all VRAM."""

    def __init__(self, cfg: Config):
        self.cfg              = cfg
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
        self._adapter_flag: bool | None = None  # cached "is adapter loaded?" check

    def _get_free_port(self, start_port: int) -> int:
        """Scan up to 1000 ports starting at `start_port` and return the first free one."""
        port = start_port
        while port < start_port + 1000:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except OSError:
                    # Port already in use; try the next one.
                    port += 1
        raise RuntimeError(f"No free port found in {start_port}..{start_port + 999}")

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
            
        if not adapter_path:
            raise ValueError("BetaZero requires an adapter_path.")

        tactic_path = os.path.join(adapter_path, "tactic")
        skeleton_path = os.path.join(adapter_path, "skeleton")

        for name, path in {
            "tactic": tactic_path,
            "skeleton": skeleton_path,
        }.items():
            if os.path.isdir(path):
                lora_modules.append(f"{name}={path}")
                self._loaded_loras.add(name)

        if not self._loaded_loras:
            raise FileNotFoundError(
                f"No adapter found in {adapter_path} "
                "(expected: tactic/ or skeleton/)"
            )
    
        if lora_modules:
            cmd += ["--enable-lora", "--max-loras", str(max(2, len(lora_modules))), "--lora-modules"] + lora_modules
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
        self._adapter_flag = None  # invalidate cached adapter state on start
        self._wait_ready(self.ready_timeout)

    def kill(self):
        """Kill the subprocess and its entire process group; the OS reclaims VRAM."""
        if self.proc:
            try:
                if self.proc.poll() is None:
                    print(f"[vLLM] Killing process group {self.proc.pid}...")
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.proc.wait()
            finally:
                self.proc = None

        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        
        self._adapter_flag = None  # invalidate cached adapter state on shutdown
        
        time.sleep(2)

    def sample(
        self,
        states: list[ProofState],
        action_type: str,
        n: int,
        *,
        prompts: list[str] | None = None,
    ) -> list[list[dict]]:
        """Sample `n` completions per state. Returns one list per state with `n` items each."""
        if not states or n <= 0:
            return [[] for _ in states]
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]
        elif len(prompts) != len(states):
            raise ValueError("prompts length must match states")
        
        lora_name = action_type
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
            return {
                "text": c["text"].strip(),
                "finish_reason": c.get("finish_reason", ""),
            }

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
