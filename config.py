from __future__ import annotations
import os
from dataclasses import dataclass, asdict, field
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

    # Training loop
    total_iterations: int = 1000
    theorems_per_iter: int = 16
    checkpoint_every: int = 50
    checkpoint_dir: str = "checkpoints"

    # Dataset
    dataset_dir: str = "problems/miniF2F-Valid"

    # Logging
    log_dir: str = "runs"

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
