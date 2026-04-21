from __future__ import annotations
import os
from dataclasses import dataclass, asdict, field
from typing import Optional
import yaml


@dataclass
class Config:
    run_name: str = "baseline"

    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    device: str = "cuda"

    # Rollout
    K: int = 32
    max_depth: int = 5
    max_nodes: int = 128
    lean_workers: int = 4
    lean_timeout: int = 60

    # GRPO
    lr: float = 1e-5
    eps_clip: float = 0.2
    beta_kl: float = 0.01
    grpo_epochs: int = 1
    mini_batch_size: int = 8
    # Micro-batches per optimizer.step (1 = disabled). Effective batch ≈ mini_batch_size * this.
    grpo_accumulation_steps: int = 1

    # Training loop
    total_iterations: int = 1000
    theorems_per_iter: int = 16
    checkpoint_every: int = 50
    checkpoint_dir: str = "outputs/checkpoints"
    # 0 = train on this iter only (no cross-iter buffer). >0 = accumulate rollout samples until >= this count before GRPO.
    min_samples_for_grpo: int = 0

    # Dataset
    dataset_dir: str = "problems/miniF2F-Valid"

    # LoRA (PEFT)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None  # None → use model default
    
    base_lora_tactic: str = "qwen_lora_tactic"
    base_lora_skeleton: str = "qwen_lora_skeleton"

    # Training efficiency
    gradient_checkpointing: bool = True
    logprob_chunk_size: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.7

    # vLLM subprocess (rollout phase)
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.5
    max_model_len: int = 2048
    vllm_ready_timeout: int = 300  # first HF download + load often exceeds 180s

    # Logging
    log_dir: str = "outputs/runs"
    rollout_graph_log_dir: Optional[str] = None  # e.g. outputs/rollouts → JSON per theorem/iter

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @property
    def run_log_dir(self) -> str:
        return os.path.join(self.log_dir, self.run_name)

    @property
    def run_checkpoint_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.run_name)
