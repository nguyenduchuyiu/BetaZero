# File: betazero/__init__.py


# File: betazero/train.py
from __future__ import annotations

import os
import gc
import torch
from tqdm import tqdm

from betazero.utils.config import Config
from betazero.utils.dataloader import TheoremDataset
from betazero.utils.graph_logger import GraphLogger
from betazero.utils.logger import setup as setup_logger
from betazero.policy.vllm_server import VLLMServer
from betazero.policy.trainable_policy import TrainablePolicy
from betazero.env.lean_env import LeanEnv
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.search import Sorrifier
from betazero.search import RewardCalculator
from betazero.search import LevelwiseRollout
from betazero.search import GRPOTrainer


def train(cfg: Config = Config()):
    log_dir  = cfg.run_log_dir
    ckpt_dir = cfg.run_checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, writer = setup_logger(log_dir)
    cfg.save(log_dir)
    logger.info(f"Run: {cfg.run_name}  |  {cfg}")

    scheduler = Lean4ServerScheduler(max_concurrent_requests=cfg.lean_workers,
                                     timeout=cfg.lean_timeout, name=cfg.run_name)
    lean      = LeanEnv(scheduler)
    sorrifier = Sorrifier(scheduler)
    reward    = RewardCalculator()
    dataset   = TheoremDataset(cfg.dataset_dir)
    trainer   = GRPOTrainer(
        lr=cfg.lr, eps_clip=cfg.eps_clip, beta_kl=cfg.beta_kl,
        grpo_epochs=cfg.grpo_epochs, mini_batch_size=cfg.mini_batch_size,
        accumulation_steps=cfg.grpo_accumulation_steps,
    )
    vllm      = VLLMServer(cfg)

    adapter_path: str | None = None  # updated each iteration
    grpo_buffer: list = []
    self_correction_buffer: list = []

    try:
        for iteration in tqdm(range(1, cfg.total_iterations + 1), desc=cfg.run_name):
            last_iter = iteration == cfg.total_iterations
            theorems = dataset.sample(cfg.theorems_per_iter)

            samples = []
            self_correction_samples: list = []
            try:
                # ── Phase 1: Rollout with vLLM subprocess ─────────────────────
                vllm.start(adapter_path)
                for j, thm in enumerate(theorems):
                    rollout = LevelwiseRollout(
                        vllm, lean, sorrifier, reward,
                        K=cfg.K, max_depth=cfg.max_depth, max_nodes=cfg.max_nodes,
                    )
                    batch, g, qv = rollout.rollout(thm)
                    samples.extend(batch)
                    self_correction_samples.extend(rollout.self_correction_buffer)
                    if cfg.rollout_graph_log_dir:
                        path = os.path.join(
                            cfg.rollout_graph_log_dir, cfg.run_name, f"iter{iteration:04d}_thm{j:02d}.json"
                        )
                        GraphLogger().save_json(g, thm, qv, filepath=path)
            finally:
                vllm.kill()
                gc.collect()
                torch.cuda.empty_cache()

            if cfg.min_samples_for_grpo > 0:
                grpo_buffer.extend(samples)
                self_correction_buffer.extend(self_correction_samples)
                train_batch = grpo_buffer
                aux_train_batch = self_correction_buffer
                need = cfg.min_samples_for_grpo
                do_train = len(train_batch) >= need or (last_iter and len(train_batch) > 0)
            else:
                train_batch = samples
                aux_train_batch = self_correction_samples
                need = 1
                do_train = len(train_batch) > 0

            logger.info(
                f"[{iteration:4d}] rollout: {len(samples)} samples  "
                f"buffer={len(grpo_buffer) if cfg.min_samples_for_grpo > 0 else len(samples)}"
                f"{'/' + str(need) if cfg.min_samples_for_grpo > 0 else ''}"
            )

            m = {
                "loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
                "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0,
            }
            if do_train:
                policy = TrainablePolicy(cfg, adapter_path)
                try:
                    m = trainer.update(policy, train_batch, aux_train_batch)
                    adapter_path = os.path.join(ckpt_dir, f"iter{iteration:04d}")
                    policy.save(adapter_path)
                finally:
                    policy.unload()
                    del policy
                    gc.collect()
                    torch.cuda.empty_cache()
                if cfg.min_samples_for_grpo > 0:
                    grpo_buffer.clear()
                    self_correction_buffer.clear()
            elif cfg.min_samples_for_grpo > 0:
                logger.info(
                    f"[{iteration:4d}] skip GRPO: buffer {len(grpo_buffer)}/{cfg.min_samples_for_grpo}"
                )

            logger.info(
                f"[{iteration:4d}]  loss={m['loss']:.4f}  kl={m['kl']:.4f}  "
                f"r_env={m['r_env_mean']:.3f}  Q={m['Q_mean']:.3f}  "
                f"solve={m['solve_rate']:.2%}  samples={m['n_samples']}  groups={m['n_groups']}"
            )
            writer.add_scalar("train/loss",       m["loss"],       iteration)
            writer.add_scalar("train/kl",         m["kl"],         iteration)
            writer.add_scalar("train/r_env_mean", m["r_env_mean"], iteration)
            writer.add_scalar("train/Q_mean",     m["Q_mean"],     iteration)
            writer.add_scalar("train/solve_rate", m["solve_rate"], iteration)
            writer.add_scalar("train/n_samples",  m["n_samples"],  iteration)

            if do_train and iteration % cfg.checkpoint_every == 0:
                logger.info(f"Checkpoint at iter {iteration}: {adapter_path}")
    finally:
        writer.close()
        scheduler.close()


# File: betazero/core/__init__.py
from .nodes import Action, NodeStatus, ProofState

__all__ = [
    "Action",
    "NodeStatus",
    "ProofState",
]

# File: betazero/core/nodes.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

NodeStatus = Literal["OPEN", "SOLVED", "FAILED"]


@dataclass(frozen=True)
class ProofState:
    """OR-node: a proof state (context, goal) in the AND/OR search graph."""
    context: str
    goal: str
    header: str = ""  # import lines from the source .lean file

    def __str__(self) -> str:
        return f"{self.context}\n⊢ {self.goal}" if self.context else f"⊢ {self.goal}"


@dataclass(frozen=True)
class Action:
    """AND-node: tactic or skeleton. `content` is raw LLM output; Lean execution uses extracted ```lean4``` body."""
    action_type: Literal["tactic", "skeleton"]
    content: str
    children: tuple[ProofState, ...] = field(default_factory=tuple)
    prompt: str = ""  # exact prompt shown to the LLM for this content

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))


# File: betazero/utils/dataloader.py
import os
import re
import random

from betazero.core.nodes import ProofState


def _extract_params(params_str: str) -> list[str]:
    """Extract inner content of each top-level bracket group."""
    groups, depth, start = [], 0, -1
    for i, c in enumerate(params_str):
        if c in "([{" and depth == 0:
            depth, start = 1, i + 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and start != -1:
                inner = params_str[start:i].strip()
                if inner:
                    groups.append(inner)
                start = -1
    return groups


def parse_lean_file(path: str) -> ProofState | None:
    """Parse a miniF2F .lean file into a ProofState."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'\btheorem\s+\w+(.*?):=\s*by\s+sorry', content, re.DOTALL)
    if not m:
        return None
    header = content[:m.start()].strip()
    sig = m.group(1).strip()
    depth, colon_pos = 0, -1
    for i, c in enumerate(sig):
        if c in "([{":   depth += 1
        elif c in ")]}": depth -= 1
        elif c == ":" and depth == 0 and (i + 1 >= len(sig) or sig[i + 1] != "="):
            colon_pos = i
    if colon_pos == -1:
        return None
    goal    = sig[colon_pos + 1:].strip()
    context = "\n".join(_extract_params(sig[:colon_pos]))
    return ProofState(context=context, goal=goal, header=header)


class TheoremDataset:
    def __init__(self, directory: str):
        paths = sorted(f for f in os.listdir(directory) if f.endswith(".lean"))
        self.theorems: list[ProofState] = []
        for name in paths:
            ps = parse_lean_file(os.path.join(directory, name))
            if ps:
                self.theorems.append(ps)
        print(f"Loaded {len(self.theorems)} theorems from {directory}")

    def sample(self, n: int) -> list[ProofState]:
        return random.choices(self.theorems, k=n)


# File: betazero/utils/config.py
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


# File: betazero/utils/graph_logger.py
import json
import os
from typing import Any, Dict

from betazero.core import ProofState, Action
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph

class GraphLogger:
    """Crawler đồ thị AND/OR để export ra JSON dùng cho visualization."""

    def __init__(self):
        self.obj_to_id: Dict[Any, str] = {}
        self.state_counter = 0
        self.action_counter = 0

    def _get_state_id(self, state: ProofState) -> str:
        if state not in self.obj_to_id:
            self.obj_to_id[state] = f"state_{self.state_counter}"
            self.state_counter += 1
        return self.obj_to_id[state]

    def _get_action_id(self, action: Action) -> str:
        if action not in self.obj_to_id:
            self.obj_to_id[action] = f"action_{self.action_counter}"
            self.action_counter += 1
        return self.obj_to_id[action]

    def export_to_dict(self, graph: ANDORGraph, root: ProofState, q_values: Dict[Action, float]) -> Dict[str, Any]:
        """Duyệt đồ thị và build cấu trúc Flat (nodes, edges)."""
        nodes = []
        edges = []
        
        visited_states = set()
        visited_actions = set()

        def traverse_state(state: ProofState):
            if state in visited_states:
                return
            visited_states.add(state)
            
            s_id = self._get_state_id(state)
            nodes.append({
                "id": s_id,
                "type": "OR",
                "status": graph.status(state),
                "depth": graph.get_depth(state),
                "content": {
                    "context": state.context,
                    "goal": state.goal
                }
            })

            for action in graph.get_actions(state):
                a_id = self._get_action_id(action)
                edges.append({"source": s_id, "target": a_id, "relation": "expanded_to"})
                traverse_action(action)

        def traverse_action(action: Action):
            if action in visited_actions:
                return
            visited_actions.add(action)

            a_id = self._get_action_id(action)
            # Truy cập internal _r_dep (nếu ông chưa viết hàm get_r_dep trong ANDORGraph)
            r_d = graph._r_dep.get(action, 0.0) 
            prompt = action.prompt or ""
            extracted_lean_code = get_lean_code(prompt.replace("\u00a0", " "))

            nodes.append({
                "id": a_id,
                "type": "AND",
                "action_type": action.action_type,
                "status": graph.status(action),
                "content": action.content,
                "prompt": action.prompt,
                "extracted_lean_code": extracted_lean_code,
                "metrics": {
                    "r_env": graph.get_r_env(action),
                    "r_dep": r_d,
                    "Q_value": q_values.get(action, 0.0)
                }
            })

            for child_state in action.children:
                c_id = self._get_state_id(child_state)
                edges.append({"source": a_id, "target": c_id, "relation": "subgoal"})
                traverse_state(child_state)

        # Bắt đầu duyệt từ root
        traverse_state(root)

        return {
            "theorem_goal": root.goal,
            "root_id": self._get_state_id(root),
            "total_nodes": len(nodes),
            "nodes": nodes,
            "edges": edges
        }

    def save_json(self, graph: ANDORGraph, root: ProofState, q_values: Dict[Action, float], filepath: str):
        """Export và lưu thành file JSON."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = self.export_to_dict(graph, root, q_values)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# File: betazero/utils/__init__.py
from .config import Config
from .dataloader import TheoremDataset
from .logger import setup as setup_logger

__all__ = [
    "Config",
    "TheoremDataset",
] 


# File: betazero/utils/logger.py
import logging
import os
from torch.utils.tensorboard import SummaryWriter


def setup(log_dir: str) -> tuple[logging.Logger, SummaryWriter]:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("betazero")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)

    writer = SummaryWriter(log_dir=log_dir)
    return logger, writer


# File: betazero/policy/trainable_policy.py
import os
import gc
import torch
from collections import defaultdict
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from betazero.core import ProofState
from betazero.utils import Config


class TrainablePolicy:
    """PEFT model for log_probs + gradient update. Explicitly load/unload to free VRAM."""

    def __init__(self, cfg: Config, adapter_path: str | None = None):
        self.device = cfg.device
        self.logprob_chunk_size = max(1, cfg.logprob_chunk_size)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, device_map=cfg.device
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()

        if adapter_path and os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        else:
            self.model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules, bias="none",
            ))
        self.model.print_trainable_parameters()

    def parameters(self):
        return self.model.parameters()

    def save(self, path: str):
        self.model.save_pretrained(path)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def _score_chunk(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        chosen = torch.gather(logits, dim=2, index=targets.unsqueeze(-1)).squeeze(-1).float()
        norm = torch.logsumexp(logits.float(), dim=-1)
        return chosen - norm

    def log_probs(self, states: list[ProofState], actions: list[str],
                  prompts: list[str], disable_adapter: bool = False) -> torch.Tensor:
        """
        Tính log_probs tối ưu bằng Shared Prefill. 
        Tự động nhóm các action có cùng prompt để chỉ prefill 1 lần.
        """
        if not prompts:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        # Gom nhóm index theo prompt
        prompt_to_idxs = defaultdict(list)
        for i, p in enumerate(prompts):
            prompt_to_idxs[p].append(i)

        scores = torch.zeros(len(prompts), dtype=torch.float32, device=self.device)
        
        ctx = self.model.disable_adapter() if disable_adapter else nullcontext()
        was_gc = bool(getattr(self.model, "is_gradient_checkpointing", False))

        if was_gc:
            self.model.gradient_checkpointing_disable()

        try:
            with ctx:
                # Xử lý tối ưu cho từng cụm prompt
                for p_text, idxs in prompt_to_idxs.items():
                    group_actions = [actions[i] for i in idxs]
                    
                    # Gọi hàm Shared Prefill
                    group_scores = self._shared_prefill_log_probs(p_text, group_actions)
                    
                    # Gán điểm trả về đúng index ban đầu
                    for i, score in zip(idxs, group_scores):
                        scores[i] = score
        finally:
            if was_gc:
                self.model.gradient_checkpointing_enable()

        return scores

    def _shared_prefill_log_probs(self, prompt: str, actions: list[str]) -> list[torch.Tensor]:
        """True Batch Parallelism: Xử lý nhiều action cùng lúc bằng sức mạnh GPU."""
        # 1. Prefill Prompt 1 lần
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prefill = self.model(input_ids=prompt_ids, use_cache=True, return_dict=True)
        
        # KV Cache của prompt (được broadcast tự động cho batch phía sau)
        prompt_pkv = prefill.past_key_values 
        last_prompt_logit = prefill.logits[:, -1:, :] # (1, 1, V)

        # 2. Tokenize toàn bộ actions với Padding
        # Tắt add_special_tokens để không bị dính BOS token thừa
        encoded_actions = self.tokenizer(actions, padding=True, add_special_tokens=False, return_tensors="pt").to(self.device)
        act_ids = encoded_actions.input_ids # (B, T_max)
        act_mask = encoded_actions.attention_mask # (B, T_max)
        
        B, T_max = act_ids.shape
        
        # Mở rộng KV Cache từ (1, ...) thành (B, ...) để khớp với batch actions
        # Lưu ý: Tùy phiên bản transformers, pkv có thể là tuple. 
        # DeepSeek/Llama thường dùng DynamicCache hoặc tuple of tuples.
        batch_pkv = self._expand_pkv(prompt_pkv, B)

        # 3. Tính Log Prob cho token đầu tiên của cả batch
        # last_prompt_logit: (1, 1, V) -> broadcast thành (B, 1, V)
        first_token_ids = act_ids[:, :1] # (B, 1)
        # Logits tại vị trí cuối prompt dùng để dự đoán token index 0 của action
        first_lp = self._score_batch_chunk(last_prompt_logit.expand(B, -1, -1), first_token_ids) # (B)
        # Nếu action rỗng (chỉ có padding), mask nó về 0
        first_lp = first_lp * act_mask[:, 0]

        # 4. Chunking dọc theo chiều dài chuỗi (T_max) nhưng chạy song song theo Batch (B)
        total_rest_lp = torch.zeros(B, device=self.device)
        
        if T_max > 1:
            teacher_inputs = act_ids[:, :-1]
            teacher_targets = act_ids[:, 1:]
            target_mask = act_mask[:, 1:] # Mask cho các token thực tế (không phải padding)
            
            past = batch_pkv
            for start in range(0, T_max - 1, self.logprob_chunk_size):
                end = start + self.logprob_chunk_size
                
                chunk_inputs = teacher_inputs[:, start:end]
                chunk_targets = teacher_targets[:, start:end]
                chunk_mask = target_mask[:, start:end]

                out = self.model(
                    input_ids=chunk_inputs,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True
                )
                
                # Tính score cho cả chunk của cả batch
                # out.logits shape: (B, chunk_len, V)
                chunk_lps = self._score_batch_chunk(out.logits, chunk_targets) # (B, chunk_len)
                total_rest_lp += (chunk_lps * chunk_mask).sum(dim=1)
                
                past = out.past_key_values

        final_scores = first_lp + total_rest_lp
        return [s for s in final_scores]

    def _score_batch_chunk(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính log_probs cho tensor (B, T, V)."""
        # logits: (B, T, V), targets: (B, T)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    def _expand_pkv(self, pkv, B):
        """Mở rộng KV Cache cho Batch Size mới."""
        if pkv is None: return None
        # pkv thường là tuple của (key, value) cho mỗi layer
        new_pkv = []
        for layer_k, layer_v in pkv:
            # layer_k: (1, num_heads, seq_len, head_dim)
            new_pkv.append((
                layer_k.expand(B, -1, -1, -1),
                layer_v.expand(B, -1, -1, -1)
            ))
        return tuple(new_pkv)

# File: betazero/policy/output_parser.py
import re
import textwrap

_LEAN_HEADER = re.compile(r"(?is)\b(theorem|lemma|example|def)\b")
_PROOF_DIVIDER = re.compile(r"(?is):=\s*by|(?<=\s)by(?=\s)")

def get_lean_code(raw: str) -> str:
    """
    Only passes through fully valid Lean actions.
    Giữ nguyên 100% khoảng trắng và lề gốc của proof body.
    """
    t = raw.strip()

    # Reject if ChatML tokens are present.
    if "<|im_" in t:
        return ""

    # Capture last ```lean4 ... ``` code block.
    # Lưu ý: Giữ lại \s+ để tương thích với regex gốc của ông
    fences = re.findall(r"```lean4\s+(.*?)\s+```", t, re.DOTALL | re.IGNORECASE)
    if not fences:
        return ""
    
    # KHÔNG .strip() code_block vội, cứ để nguyên raw string
    code_block = fences[-1]

    # Require header and proof divider.
    if not _LEAN_HEADER.search(code_block) or not _PROOF_DIVIDER.search(code_block):
        return ""

    divider_match = _PROOF_DIVIDER.search(code_block)
    
    # Lấy từ vị trí ngay sau chữ 'by' trở đi, giữ nguyên mọi dấu \n và space
    proof_body = code_block[divider_match.end():]

    if not proof_body or "<|im_" in proof_body:
        return ""

    return proof_body
    
if __name__ == "__main__":
    test = '''- After change: `(25 + 5) rows * (18 - 3) seats = 30 rows * 15 seats = 450 seats`.
    - Both arrangements result in 450 seats, confirming the solution is correct.

**Lean 4 Code:**

```lean4
theorem <ACTUAL_THEOREM_NAME> (<ACTUAL_VARIABLES>) : <ACTUAL_GOAL> := by
    -- Use the quadratic formula to solve the quadratic equation derived from the problem.
    have h : rows = 25 := by
        -- Apply the quadratic formula to the equation rows^2 + 5rows - 750 = 0.
        apply Eq.symm
        nlinarith [sq_nonneg (rows - 25), sq_nonneg (rows + 30)]
    -- Substitute the value of rows into the equation to find the number of seats.
    have h' : seats = 18 := by
        subst h
        nlinarith
    -- Verify the solution by checking the original and modified arrangements.
    subst_vars
    nlinarith
```'''
    print(get_lean_code(test))

# File: betazero/policy/prompt.py
from __future__ import annotations
import textwrap
from betazero.core.nodes import ProofState

_SYSTEM_BASE_INSTRUCTION = textwrap.dedent("""\
You are an elite expert in Lean 4 theorem proving.
Your task is to advance or solve the given mathematical state.
""").strip()

_USER_BASE_INSTRUCTION = textwrap.dedent("""\
This is the current state of the proof.
You may only use the information in the following [CONTEXT] and [GOAL].
""").strip()

_TACTIC_INSTRUCTION = textwrap.dedent("""\
Write Lean 4 tactics to solve the goal based on the provided [CONTEXT] and [GOAL].

### EXAMPLES OF EXPECTED OUTPUT

Example 1:

[CONTEXT]
p q : Prop
hp : p
hq : q

[GOAL]
p ∧ q

# Example 1 OUTPUT

<think>
The goal is a conjunction. I have both individual components in the context. I can use the constructor tactic.
</think>
```lean4
theorem solve_conj (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor
  · exact hp
  · exact hq
```
Example 2:

[CONTEXT]
n : ℕ

[GOAL]
n + 0 = n

# Example 2 OUTPUT
<think>
This is a fundamental property of natural numbers in Lean. The rw tactic with Nat.add_zero will solve it.
</think>
```lean4
theorem add_zero_id (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
```
""").strip()

_SKELETON_INSTRUCTION = textwrap.dedent("""\
Write a modular Lean 4 proof skeleton. Break the goal into logical intermediate steps using 'have'.
### EXAMPLES OF EXPECTED OUTPUT

Example 1:

[CONTEXT]
a b c : ℕ

[GOAL]
(a + b) + c = a + (b + c)

# Example 1 OUTPUT
<think>
I will prove this by showing the left side equals the right side using associativity.
</think>
```lean4
theorem add_assoc_skeleton (a b c : ℕ) : (a + b) + c = a + (b + c) := by
  have h1 : (a + b) + c = a + b + c := by
    sorry
  have h2 : a + (b + c) = a + b + c := by
    sorry
  rw [h1, h2]
```

Example 2:

[CONTEXT]
p q : Prop

[GOAL]
p ↔ q

# Example 2 OUTPUT
<think>
An iff goal requires proving two directions: p → q and q → p.
</think>
```lean4
theorem iff_skeleton (p q : Prop) : p ↔ q := by
  constructor
  · intro hp
    sorry
  · intro hq
    sorry
```
""").strip()

# Rollout splits phase-2 tactic samples when this appears in `Action.prompt`.
TACTIC_SELF_CORRECT_USER_MARKER = "[PREVIOUS FAILED TACTIC]"

_TACTIC_SELF_CORRECT_INSTRUCTION = textwrap.dedent("""\
Analyze the compiler feedback and the failed tactic. Write a complete Lean 4 tactic that fixes the error.
You should use the [SYNTAX-FIXED REFERENCE] as a hint for the correct structure.

### EXAMPLES OF SELF-CORRECTION

Example 1 (Error: Unknown identifier):

[CONTEXT]
n : ℕ

[GOAL]
n + 0 = n

[PREVIOUS FAILED TACTIC]
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  rw [add_zero_property]
```
  
[LEAN 4 COMPILER FEEDBACK]
error: unknown identifier 'add_zero_property'

[SYNTAX-FIXED REFERENCE (Used sorry)]
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  sorry
```

# Example 1 OUTPUT
<think>
The compiler can't find 'add_zero_property'. Looking at the syntax-fixed reference, the correct lemma name is 'Nat.add_zero'.
</think>
```lean4
theorem add_zero_fix (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
```
""").strip()

def _format_chatml(system_msg: str, user_msg: str) -> str:
    full_prompt = (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return clean_prompt(full_prompt)

def build_prompt(state: ProofState, action_type: str) -> str:
    if action_type == "tactic":
        instruction = _TACTIC_INSTRUCTION
    elif action_type == "skeleton":
        instruction = _SKELETON_INSTRUCTION
    else:
        raise ValueError(action_type)

    full_system = _SYSTEM_BASE_INSTRUCTION + '\n\n' + instruction
    user_msg = (
        _USER_BASE_INSTRUCTION + "\n\n" +
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip()
    )
    return _format_chatml(full_system, user_msg)

def build_tactic_self_correct_prompt(
    state: ProofState,
    original_tactic: str,
    lean_feedback: str,
    sorrified_tactic: str,
    ) -> str:
    
    full_system = _SYSTEM_BASE_INSTRUCTION + '\n\n' + _TACTIC_SELF_CORRECT_INSTRUCTION
    user_msg = (
        _USER_BASE_INSTRUCTION + "\n\n" +
        "[CONTEXT]\n" + state.context.strip() + "\n\n" +
        "[GOAL]\n" + state.goal.strip() + "\n\n" +
        f"{TACTIC_SELF_CORRECT_USER_MARKER}\n```lean4\n" + original_tactic.strip() + "\n```\n\n" +
        "[LEAN 4 COMPILER FEEDBACK]\n" + (lean_feedback.strip() or '(No clear error message)') + "\n\n" +
        "[SYNTAX-FIXED REFERENCE (Used sorry)]\n```lean4\n" + sorrified_tactic.strip() + "\n```"
    )
    return _format_chatml(full_system, user_msg)

def clean_prompt(text: str) -> str:
    # Thay thế NBSP (khoảng trắng lạ) bằng khoảng trắng chuẩn ASCII 32
    return text.replace('\u00a0', ' ')

# if __name__ == "__main__":
#     state = ProofState(
#         context="n : ℕ",
#         goal="n + 0 = n",
#     )
#     original_tactic = "theorem add_zero_fix (n : ℕ) : n + 0 = n := by\n  rw [add_zero_property]"
#     lean_feedback = "error: unknown identifier 'add_zero_property'"
#     sorrified_tactic = "theorem add_zero_fix (n : ℕ) : n + 0 = n := by\n  sorry"
#     print("-" * 100)
#     print("SELF-CORRECT PROMPT")
#     print("-" * 100)
#     print(build_tactic_self_correct_prompt(state, original_tactic, lean_feedback, sorrified_tactic))
#     print("-" * 100)
#     print("SKELETON PROMPT")
#     print("-" * 100)
#     print(build_prompt(state, "skeleton"))
#     print("-" * 100)
#     print("TACTIC PROMPT")
#     print("-" * 100)
#     print(build_prompt(state, "tactic"))
#     print("-" * 100)

# File: betazero/policy/__init__.py
from .trainable_policy import TrainablePolicy
from .vllm_server import VLLMServer

__all__ = [
    "TrainablePolicy",
    "VLLMServer",
]   


# File: betazero/policy/vllm_server.py
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
        
        self.port = self._get_free_port(self.base_port)
        env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_util),
            "--max-model-len", str(self.max_model_len),
        ]
        if adapter_path and os.path.exists(adapter_path):
            cmd += ["--enable-lora", "--lora-modules", f"adapter={adapter_path}"]
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
    ) -> list[list[str]]:
        """Sample `n` completions per state row. Returns len(states) lists, each of length `n`."""
        if not states or n <= 0:
            return [[] for _ in states]
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]
        elif len(prompts) != len(states):
            raise ValueError("prompts length must match states")
        
        # Use cached adapter flag to avoid HTTP request per sample call.
        if self._adapter_flag is None:
            self._adapter_flag = self._adapter_loaded()
        model = "adapter" if self._adapter_flag else self.model_name
        try:
            r = requests.post(
                f"http://localhost:{self.port}/v1/completions",
                json={"model": model, "prompt": prompts, "n": n,
                      "max_tokens": self.max_tokens, "temperature": self.temperature},
                timeout=(10, 1800),
            )
            r.raise_for_status()
            choices = r.json().get("choices", [])
        except requests.exceptions.RequestException as e:
            print(f"[vLLM] sample error: {e}")
            return [[] for _ in states]
        
        return [[choices[i * n + j]["text"].strip() for j in range(n)]
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


# File: betazero/search/__init__.py
from .graph import ANDORGraph
from .reward import DependencyRewardAssigner, RewardCalculator
from .rollout import BatchExecutor, FailureHandler, LevelwiseRollout, RolloutBudget, SamplePolicy
from .sorrifier import Sorrifier
from .trainer import GRPOTrainer

__all__ = [
    "ANDORGraph",
    "BatchExecutor",
    "DependencyRewardAssigner",
    "FailureHandler",
    "GRPOTrainer",
    "LevelwiseRollout",
    "RewardCalculator",
    "RolloutBudget",
    "SamplePolicy",
    "Sorrifier",
]


# File: betazero/search/sorrifier/sorrifier.py
"""
AST-Based Automated Proof Patcher for Lean 4
--------------------------------------------
This module automates the process of fixing broken Lean 4 proofs by replacing 
faulty tactics with the `sorry` axiom. 

Architecture:
1. AST-Guided Truncation: Uses Lean's AST (with Elaborator) to precisely locate tactic boundaries using Spatial Heuristics (Byte Length).
2. Indentation Heuristics: Infers structural hierarchy where AST lacks context (e.g., closing scopes).
3. Oscillation Fallback: Detects infinite correction loops caused by Lean's syntax 
   intolerance and resets the parent block to prevent halting.
"""

from __future__ import annotations
import sys
import datetime
from typing import Tuple, List, Dict, TextIO, Optional
from tqdm import tqdm
from betazero.env import Lean4ServerScheduler
from betazero.env.ast_parser import get_lean_ast

BLOCK_STARTERS = (
    "have", "·", ".", "cases ", "cases' ", "induction ", 
    "induction' ", "rintro ", "intro ", "calc", "match", 
    "lemma", "theorem", "def", "example"
)

TRIVIAL_TACTICS = frozenset({"skip", "done", "trivial", "decide", "rfl"})


class Sorrifier:
    def __init__(
        self,
        repl_verifier: Lean4ServerScheduler,
        max_cycles: int = 50,
        log_path: Optional[str] = None,
    ):
        self.repl_verifier = repl_verifier
        self.max_cycles = max_cycles
        self.log_path = log_path
        self.current_content = ""
        self._last_action_msg = ""
        self._log_fp: Optional[TextIO] = None

    def _log_open(self):
        if self.log_path and self._log_fp is None:
            self._log_fp = open(self.log_path, "w", encoding="utf-8")

    def _log_close(self):
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

    def _log(self, text: str) -> None:
        if self._log_fp:
            self._log_fp.write(text)
            if not text.endswith("\n"):
                self._log_fp.write("\n")
            self._log_fp.flush()

    def _log_section(self, title: str) -> None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        bar = "=" * 76
        self._log(f"\n{bar}\n[{ts}] {title}\n{bar}\n")

    @staticmethod
    def _format_numbered_source(content: str, width: int = 5) -> str:
        lines = content.splitlines()
        out = [f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1)]
        return "\n".join(out) + ("\n" if out else "")

    def _log_source_block(self, label: str, content: str, err_line: int | None = None) -> None:
        self._log(f"--- {label} (numbered) ---")
        if err_line is not None:
            self._log(f"(error line ref: L{err_line})")
        self._log(self._format_numbered_source(content))

    def _log_error_batch(
        self,
        fatal_errors: List[Tuple[int, str]],
        unsolved_goals: List[Tuple[int, str]],
        primary_line: int,
        primary_msg: str,
        is_fatal: bool,
    ) -> None:
        self._log("Primary (first) issue:")
        self._log(f"  kind: {'fatal' if is_fatal else 'unsolved_goals'}")
        self._log(f"  line: {primary_line}")
        self._log(f"  message: {primary_msg}")
        if fatal_errors:
            self._log("All fatal errors (line, message):")
            for ln, msg in fatal_errors:
                self._log(f"  L{ln}: {msg[:500]}{'…' if len(msg) > 500 else ''}")
        if unsolved_goals:
            self._log("All unsolved goals (line, message):")
            for ln, msg in unsolved_goals:
                self._log(f"  L{ln}: {msg[:500]}{'…' if len(msg) > 500 else ''}")

    def fix_code(self, code: str) -> str:
        """Iteratively patch Lean 4 errors until the code compiles or max_cycles is reached."""
        self._log_open()
        try:
            self.current_content = self._strip_noop_tactics(code)
            self._last_action_msg = ""
            seen_states = set()

            if self._log_fp:
                self._log_section("Sorrifier start")
                self._log("After _strip_noop_tactics, initial source:")
                self._log_source_block("INITIAL", self.current_content)

            with tqdm(total=self.max_cycles, desc="Processing", unit="cycle") as pbar:
                for cycle in range(1, self.max_cycles + 1):
                    try:
                        fatal_errors, unsolved_goals = self._get_lean_errors()
                    except RuntimeError as e:
                        tqdm.write(f"\nHALTED: {e}")
                        if self._log_fp:
                            self._log_section(f"HALTED (cycle {cycle}) — Lean/runtime")
                            self._log(str(e))
                            self._log_source_block("STATE AT HALT", self.current_content)
                        return self._force_full_sorrify()

                    if not fatal_errors and not unsolved_goals:
                        if self._log_fp:
                            self._log_section(f"SUCCESS (cycle {cycle})")
                            self._log("No fatal errors and no unsolved goals.")
                            self._log_source_block("FINAL", self.current_content)
                        return self.current_content

                    is_fatal = bool(fatal_errors)
                    err_line, err_msg = fatal_errors[0] if is_fatal else unsolved_goals[0]
                    if not self._is_valid_line_number(err_line):
                        try:
                            fatal_errors, unsolved_goals = self._get_lean_errors()
                            is_fatal = bool(fatal_errors)
                            err_line, err_msg = fatal_errors[0] if is_fatal else unsolved_goals[0]
                        except RuntimeError:
                            pass
                    err_line = self._normalize_line_number(err_line)

                    if self.current_content in seen_states:
                        tqdm.write(f"\nOscillation detected at line {err_line}. Triggering Parent Block Reset...")
                        if self._log_fp:
                            self._log_section(f"Cycle {cycle} — OSCILLATION / parent block reset")
                            self._log_error_batch(fatal_errors, unsolved_goals, err_line, err_msg, is_fatal)
                            self._log_source_block("BEFORE reset", self.current_content, err_line)
                        try:
                            self._resolve_infinite_loop(err_line)
                        except IndexError as e:
                            tqdm.write(f"Index error during oscillation fallback: {e}. Force full sorrify.")
                            if self._log_fp:
                                self._log_section("FORCE full sorrify (IndexError in oscillation)")
                                self._log(str(e))
                                self._log_source_block("STATE", self.current_content)
                            return self._force_full_sorrify()
                        if self._log_fp:
                            self._log("--- AFTER reset ---")
                            if self._last_action_msg:
                                self._log(f"action: {self._last_action_msg}")
                            self._log_source_block("AFTER reset", self.current_content, err_line)
                        pbar.update(1)
                        continue

                    seen_states.add(self.current_content)
                    pbar.set_postfix_str(f"{'Fatal' if is_fatal else 'Unsolved'} @ L{err_line}")

                    if self._log_fp:
                        self._log_section(f"Cycle {cycle} — normal fix")
                        self._log_error_batch(fatal_errors, unsolved_goals, err_line, err_msg, is_fatal)
                        self._log_source_block("BEFORE fix", self.current_content, err_line)

                    try:
                        success = self._apply_normal_fix(err_line, is_fatal, err_msg)
                    except IndexError as e:
                        tqdm.write(f"Index error during normal fix: {e}. Force full sorrify.")
                        if self._log_fp:
                            self._log_section("FORCE full sorrify (IndexError in normal fix)")
                            self._log(str(e))
                            self._log_source_block("STATE", self.current_content)
                        return self._force_full_sorrify()
                    if not success:
                        tqdm.write(f"\nHALTED: Unrecoverable error at line {err_line}.")
                        if self._log_fp:
                            self._log_section(f"HALTED (cycle {cycle}) — unrecoverable")
                            self._log(f"line {err_line}")
                            self._log_source_block("STATE AT HALT", self.current_content, err_line)
                        break

                    if self._log_fp:
                        self._log("--- AFTER fix ---")
                        if self._last_action_msg:
                            self._log(f"action: {self._last_action_msg}")
                        self._log_source_block("AFTER fix", self.current_content, err_line)

                    pbar.update(1)

            if self._log_fp:
                self._log_section("Run finished — loop ended without early success return")
                self._log("(Either max_cycles exhausted or break after unrecoverable error.)")
                self._log_source_block("FINAL", self.current_content)
        finally:
            self._log_close()

        return self.current_content

    # ==========================================
    # CORE FIXING LOGIC
    # ==========================================

    def _resolve_infinite_loop(self, err_line: int):
        lines = self.current_content.splitlines()
        err_line = self._normalize_line_number(err_line, total_lines=len(lines))
        
        line_str = lines[err_line - 1]
        indent = len(line_str) - len(line_str.lstrip())
        
        if any(line_str.strip().startswith(kw) for kw in ["lemma", "theorem", "def", "example"]):
            lines.append(" " * 2 + "sorry")
        else:
            # Phanh khẩn cấp xịn: Comment dòng lỗi và toàn bộ các dòng con thụt lề sâu hơn
            lines[err_line - 1] = "-- " + lines[err_line - 1]
            i = err_line
            while i < len(lines) and lines[i].strip():
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > indent:
                    lines[i] = "-- " + lines[i]
                    i += 1
                else:
                    break
            # Nhét sorry vào cuối để thoát kẹt
            lines.insert(i, " " * indent + "sorry")
            
        self.current_content = "\n".join(lines) + "\n"

    def _apply_normal_fix(self, error_line: int, is_fatal: bool, err_msg: str) -> bool:
        lines = self.current_content.splitlines()
        error_line = self._normalize_line_number(error_line, total_lines=len(lines))

        line_content = lines[error_line - 1].strip()
        if line_content in TRIVIAL_TACTICS:
            lines[error_line - 1] = ""
            self._last_action_msg = f"Removed failing trivial tactic '{line_content}' at L{error_line}"
            tqdm.write(self._last_action_msg)
            self.current_content = "\n".join(lines) + "\n"
            return True

        blocks = self._get_ast_lines()
        enclosing = [b for b in blocks if b["start_line"] <= error_line <= b["end_line"] and b["kind"] != "Module"]

        def emergency_fallback():
            indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
            line_str = lines[error_line - 1]
            if any(line_str.strip().startswith(kw) for kw in ["lemma", "theorem", "def", "example"]):
                lines.append(" " * 2 + "sorry")
            else:
                lines[error_line - 1] = "-- " + line_str + "\n" + " " * indent + "sorry"
            self.current_content = "\n".join(lines) + "\n"
            return True

        if not enclosing:
            return emergency_fallback()

        if is_fatal:
            # CHỐNG BÃO SORRY: Lean đòi Command, tuyệt đối không chèn sorry bừa bãi
            if "expected command" in err_msg.lower():
                indent = len(lines[error_line - 1]) - len(lines[error_line - 1].lstrip())
                lines[error_line - 1] = "-- " + lines[error_line - 1]
                i = error_line
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("--"):
                    curr_indent = len(lines[i]) - len(lines[i].lstrip())
                    if curr_indent > indent:
                        lines[i] = "-- " + lines[i]
                        i += 1
                    else:
                        break
                self._last_action_msg = f"Commented invalid syntax block at L{error_line}"
                self.current_content = "\n".join(lines) + "\n"
                return True

            valid_nodes = [b for b in enclosing if "command" not in b["kind"].lower()]
            if not valid_nodes: return emergency_fallback()
            
            target = min(valid_nodes, key=lambda x: x["end_byte"] - x["start_byte"])
            L_start, L_end = target["start_line"], target["end_line"]
            start_line_str = lines[L_start - 1]
            indent = len(start_line_str) - len(start_line_str.lstrip())
            
            is_orphan_error = "no goals" in err_msg.lower() or "goals accomplished" in err_msg.lower()
            
            if is_orphan_error:
                for i in range(L_start - 1, L_end):
                    lines[i] = "-- " + lines[i]
                self._last_action_msg = f"Commented orphaned tactic [{target['kind']}] L{L_start}..L{L_end}"
                self.current_content = "\n".join(lines) + "\n"
                return True

            if self._is_block_starter(start_line_str):
                new_lines = lines[:L_end]
                new_lines.append(" " * (indent + 2) + "sorry")
                new_lines.extend(lines[L_end:])
                self._last_action_msg = f"Appended sorry to block [{target['kind']}] starting at L{L_start}"
                self.current_content = "\n".join(new_lines) + "\n"
            else:
                for i in range(L_start - 1, L_end):
                    lines[i] = "-- " + lines[i]
                new_lines = lines[:L_end]
                new_lines.append(" " * indent + "sorry")
                new_lines.extend(lines[L_end:])
                self._last_action_msg = f"Commented failing tactic [{target['kind']}] L{L_start}..L{L_end}"
                self.current_content = "\n".join(new_lines) + "\n"

        else: 
            # UNSOLVED GOALS: Đã fix danh sách scope chuẩn, đéo bao giờ vồ nhầm lá nữa!
            scopes = [
                "tactichave__", "tacticcases__", "tacticlet__", 
                "tacticinduction__", "tacticcalc__", "tacticmatch__", 
                "bytactic", "declval"
            ]
            valid_nodes = [b for b in enclosing if any(s in b["kind"].lower() for s in scopes)]
            
            if not valid_nodes:
                lines.append("  sorry")
                self.current_content = "\n".join(lines) + "\n"
                return True
                
            target = min(valid_nodes, key=lambda x: x["end_byte"] - x["start_byte"])
            L_start, L_end = target["start_line"], target["end_line"]
            
            parent_indent = len(lines[L_start - 1]) - len(lines[L_start - 1].lstrip())
            indent = parent_indent + 2 
            
            for i in range(L_start, L_end):
                line = lines[i]
                if line.strip() and not line.strip().startswith("--"):
                    indent = len(line) - len(line.lstrip())
                    break

            self._last_action_msg = f"Closed scope [{target['kind']}] at L{L_end} (Indent: {indent})"
            tqdm.write(self._last_action_msg)
            
            new_lines = lines[:L_end]
            new_lines.append(" " * indent + "sorry")
            new_lines.extend(lines[L_end:])
            
            self.current_content = "\n".join(new_lines) + "\n"

        return True

    def _get_lean_errors(self) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        result = self.repl_verifier.verify(self.current_content)

        if result.get("system_errors"):
            raise RuntimeError(f"Lean verification timed out or crashed: {result['system_errors'][:200]}")

        fatal_errors: List[Tuple[int, str]] = []
        unsolved_goals: List[Tuple[int, str]] = []

        for msg in result.get("errors", []):
            ln = msg.get("pos", {}).get("line", 1)
            txt = msg.get("data", "")
            if "unsolved goals" in txt:
                unsolved_goals.append((ln, txt))
            else:
                fatal_errors.append((ln, txt))

        return sorted(fatal_errors), sorted(unsolved_goals)

    def _get_ast_lines(self) -> List[Dict]:
        blocks = get_lean_ast(self.current_content)
        raw_bytes = self.current_content.encode('utf-8')
        for b in blocks:
            b["start_line"] = self._byte_to_line(raw_bytes, b["start_byte"])
            b["end_line"] = self._byte_to_line(raw_bytes, b["end_byte"])
        return blocks

    def _clean_redundant_sorries(self, lines: List[str]) -> str:
        cleaned = []
        for line in lines:
            if line == "": continue
            stripped = line.strip()
            if stripped == "sorry" and cleaned and cleaned[-1].strip() == "sorry": continue
            cleaned.append(line)
        return "\n".join(cleaned) + "\n"

    def _force_full_sorrify(self) -> str:
        marker = ":= by"
        idx = self.current_content.find(marker)
        if idx != -1:
            prefix = self.current_content[: idx + len(marker)]
            return prefix + "\n  sorry\n"
        return self.current_content

    def _is_valid_line_number(self, line_no: int) -> bool:
        total = len(self.current_content.splitlines())
        return total > 0 and 1 <= line_no <= total

    def _normalize_line_number(self, line_no: int, total_lines: int | None = None) -> int:
        if total_lines is None: total_lines = len(self.current_content.splitlines())
        if total_lines <= 0: return 1
        return max(1, min(line_no, total_lines))

    def _normalize_line_range(self, start_line: int, end_line: int, total_lines: int) -> Tuple[int, int]:
        if total_lines <= 0: return 1, 1
        start = self._normalize_line_number(start_line, total_lines)
        end = self._normalize_line_number(end_line, total_lines)
        if end < start: end = start
        return start, end

    @staticmethod
    def _byte_to_line(raw_bytes: bytes, byte_offset: int) -> int:
        return raw_bytes[:byte_offset].count(b"\n") + 1

    @staticmethod
    def _strip_noop_tactics(code: str) -> str:
        lines = [l for l in code.splitlines() if l.strip() not in ("skip", "done")]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _is_block_starter(line: str) -> bool:
        stripped = line.strip()
        if stripped.startswith("_") and ":=" in stripped: return True
        if not any(stripped.startswith(cmd) for cmd in BLOCK_STARTERS): return False
        if stripped.startswith("have") and ":=" not in stripped: return False
        return True

# File: betazero/search/sorrifier/dependency_analyzer.py
from __future__ import annotations
from typing import Dict, List, Any

class ExprDependencyAnalyzer:
    def _contains_sorry(self, node: Any) -> bool:
        if not isinstance(node, dict): return False
        if node.get("expr") == "const" and node.get("name") in ("sorryAx", "sorry"): return True
        return any(self._contains_sorry(v) for k, v in node.items() if k != "expr")

    def _is_bvar_used(self, node: Any, target_idx: int) -> bool:
        if not isinstance(node, dict): return False
        expr_type = node.get("expr")
        if expr_type == "bvar":
            return node.get("idx") == target_idx
        if expr_type in ("lam", "forallE", "letE"):
            # Binders: check type/val (idx giữ nguyên), check body (idx + 1)
            return (self._is_bvar_used(node.get("var_type"), target_idx) or 
                    self._is_bvar_used(node.get("val"), target_idx) or 
                    self._is_bvar_used(node.get("body"), target_idx + 1))
        return any(self._is_bvar_used(v, target_idx) for k, v in node.items() if k != "expr")

    def classify_skeleton_subgoals(self, root_expr: Dict[str, Any]) -> Dict[str, List[str]]:
        results = {"core_solved": [], "core_failed": [], "malignant": [], "benign": []}

        def traverse(node: Any):
            if not isinstance(node, dict): return
            
            # CASE 1: Node là letE (Dạng have tường minh)
            if node.get("expr") == "letE":
                self._classify(node, node.get("val"), results)
            
            # CASE 2: Node là App(Lam, Val) - Dạng have bị convert
            elif node.get("expr") == "app" and isinstance(node.get("fn"), dict) and node["fn"].get("expr") == "lam":
                lam_node = node["fn"]
                val_node = node.get("arg") # Giá trị truyền vào chính là proof của subgoal
                self._classify(lam_node, val_node, results)

            for k, v in node.items():
                if k != "expr" and isinstance(v, dict): traverse(v)

        traverse(root_expr)
        return {k: list(set(v)) for k, v in results.items()}

    def _classify(self, binder_node: dict, val_node: Any, results: dict):
        var_name = binder_node.get("var_name", "")
        if var_name and not var_name.startswith("_"):
            # Biến là bvar 0 bên trong body của chính nó
            is_used = self._is_bvar_used(binder_node.get("body"), 0)
            is_failed = self._contains_sorry(val_node)
            
            if is_used:
                results["core_failed" if is_failed else "core_solved"].append(var_name)
            else:
                results["malignant" if is_failed else "benign"].append(var_name)

    def get_unused_context_variables(self, root_expr: Dict[str, Any]) -> List[str]:
        unused_vars = []
        current_node = root_expr

        # Unroll the top-level parameters of the theorem sequentially
        while isinstance(current_node, dict):
            expr_type = current_node.get("expr")
            
            if expr_type in ("forallE", "lam"):
                var_name = current_node.get("var_name", "")
                body = current_node.get("body", {})
                
                # We usually ignore internal compiler-generated variables
                if var_name and not var_name.startswith("_"):
                    # If the parameter is not used anywhere in the rest of the theorem
                    if not self._is_bvar_used(body, target_idx=0):
                        unused_vars.append(var_name)
                
                current_node = body # Tiếp tục đi sâu vào chuỗi tham số
            
            elif expr_type == "mdata":
                current_node = current_node.get("inner", {})
                
            else:
                # Đã thoát khỏi chuỗi tham số đầu vào (ví dụ gặp body chính, letE, app, ...)
                # Dừng lại, không quét tiếp vào sâu bên trong nữa để tránh bắt nhầm biến nội bộ.
                break

        return unused_vars
    
SHARED_EXPR_ANALYZER = ExprDependencyAnalyzer()

# File: betazero/search/sorrifier/__init__.py
"""Sorrifier for patching failed tactics/skeletons."""

from .sorrifier import Sorrifier

__all__ = ["Sorrifier"]


# File: betazero/search/sorrifier/stitcher.py
"""Safe textual proof stitcher for filling skeleton subgoals."""

from __future__ import annotations
import re

class ProofStitcher:
    """Stitches child proof blocks into skeleton sorry placeholders."""

    @staticmethod
    def stitch(skeleton_code: str, child_proofs: list[str | None]) -> str:
        """
        Replaces each `sorry` in the skeleton with the corresponding child proof.
        If a child proof is None (FAILED), `sorry` remains.
        """
        # Split strictly by the word 'sorry'
        parts = re.split(r'\bsorry\b', skeleton_code)
        
        if len(parts) - 1 != len(child_proofs):
            # Fallback for LLM hallucination: mismatch between sorry count and children
            return skeleton_code

        stitched = parts[0]
        for i, proof in enumerate(child_proofs):
            if proof is not None:
                # Calculate base indentation from the line containing the 'sorry'
                lines = parts[0].splitlines()
                indent = " " * (len(lines[-1]) - len(lines[-1].lstrip())) if lines else ""
                
                # Indent child proof lines appropriately
                proof_lines = proof.splitlines()
                indented_proof = "\n".join(
                    (indent + l if idx > 0 else l) for idx, l in enumerate(proof_lines)
                )
                stitched += indented_proof
            else:
                stitched += "sorry"
                
            stitched += parts[i + 1]

        return stitched

# File: betazero/search/rollout/execution_result.py
from __future__ import annotations

from dataclasses import dataclass, field

from betazero.core import ProofState


@dataclass(frozen=True)
class LeanExecutionResult:
    """Outcome of one Lean verify run: either a normal verify dict or a wrapped transport error."""

    state_code: str
    verify: dict
    subgoals: tuple[ProofState, ...] = field(default_factory=tuple)

    @classmethod
    def ok(cls, state_code: str, verify: dict, subgoals: list[ProofState]) -> LeanExecutionResult:
        return cls(state_code=state_code, verify=dict(verify), subgoals=tuple(subgoals))

    @classmethod
    def from_transport_error(cls, message: str, state_code: str = "") -> LeanExecutionResult:
        return cls(
            state_code=state_code,
            verify={
                "complete": False,
                "pass": False,
                "errors": [],
                "warnings": [],
                "sorries": [],
                "system_errors": message,
            },
            subgoals=(),
        )

    @property
    def system_errors(self) -> str:
        return str(self.verify.get("system_errors") or "").strip()

    @property
    def has_system_failure(self) -> bool:
        return bool(self.system_errors)


# File: betazero/search/rollout/levelwise_rollout.py
from __future__ import annotations

from typing import Protocol

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.prompt import (
    TACTIC_SELF_CORRECT_USER_MARKER,
    build_prompt,
    build_tactic_self_correct_prompt,
)
from betazero.search.graph import ANDORGraph
from betazero.search.reward import DependencyRewardAssigner, RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler


class SamplePolicy(Protocol):
    """`n` = completions per state; return[i] has up to `n` strings for states[i]."""

    def sample(
        self, states: list[ProofState], action_type: str, n: int, *, prompts: list[str] | None = None
    ) -> list[list[str]]: ...


class LevelwiseRollout:
    """Runs level-wise tactic and skeleton rollout over the proof graph under a node budget."""

    def __init__(
        self,
        policy: SamplePolicy,
        lean: LeanEnv,
        sorrifier: Sorrifier,
        reward: RewardCalculator,
        K: int = 32,
        max_depth: int = 5,
        max_nodes: int = 128,
        tactic_ratio: float = 0.8,
        *,
        executor: BatchExecutor | None = None,
        failure_handler: FailureHandler | None = None,
        reward_assigner: DependencyRewardAssigner | None = None,
    ):
        assert K >= 1, "K must be at least 1"
        self.policy = policy
        self.lean = lean
        self.reward = reward
        self.K_tac = max(1, int(K * tactic_ratio))
        self.K_skel = K - self.K_tac
        self.max_depth = max_depth
        self._budget = RolloutBudget(max_nodes)

        if executor is None:
            self.failure_handler = failure_handler or FailureHandler(lean, sorrifier, reward)
            self.executor = BatchExecutor(lean, self.failure_handler, reward)
        else:
            self.executor = executor
            self.failure_handler = failure_handler
            
        self.reward_assigner = reward_assigner or DependencyRewardAssigner(lean, reward)
        self.self_correction_buffer: list[tuple[ProofState, Action, float, float]] = []

    @property
    def max_nodes(self) -> int:
        return self._budget.max_nodes

    @property
    def total_expanded(self) -> int:
        return self._budget.used

    def rollout(self, theorem: ProofState) -> list[tuple[ProofState, Action, float, float]]:
        self.self_correction_buffer = []
        graph = ANDORGraph(theorem)
        for depth in range(self.max_depth):
            frontier = [s for s in graph.unsolved_states() if graph.get_depth(s) == depth]
            if not frontier or self._budget.used >= self._budget.max_nodes:
                break

            self._run_tactic_phase(graph, frontier)

            skel_frontier = [s for s in frontier if not graph.is_solved(s)]
            if skel_frontier and self.K_skel > 0 and self._budget.used < self._budget.max_nodes:
                self._run_skeleton_phase(graph, skel_frontier)

        self.reward_assigner.assign(graph)
        q_values = self.reward.compute_returns(graph)
        samples: list[tuple[ProofState, Action, float, float]] = []
        for a, q in q_values.items():
            tup = (graph.get_parent(a, theorem), a, graph.get_r_env(a), q)
            if TACTIC_SELF_CORRECT_USER_MARKER in a.prompt:
                self.self_correction_buffer.append(tup)
            else:
                samples.append(tup)
        return samples, graph, q_values

    def _run_tactic_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        """
        Executes a two-stage tactic rollout with a self-correction loop.
        First attempt is exploratory; the second attempt uses feedback from 
        failed attempts to refine the policy's output.
        """
        # Split the tactic budget into two rounds
        first_round_budget = max(1, self.K_tac // 2)
        first_prompts = [build_prompt(s, "tactic") for s in frontier]
        first_round_actions = self.policy.sample(
            frontier, "tactic", first_round_budget, prompts=first_prompts
        )
        round_one_outcomes = self.executor.execute(
            graph, frontier, first_round_actions, "tactic", self._budget, prompts=first_prompts
        )

        # Prepare self-correction data for states that were not solved in Round 1
        correction_states: list[ProofState] = []
        correction_prompts: list[str] = []
        
        for state, per_action in zip(frontier, round_one_outcomes):
            if graph.is_solved(state):
                continue
            for feedback in per_action:
                if feedback is None:
                    continue
                correction_states.append(state)
                correction_prompts.append(build_tactic_self_correct_prompt(state, *feedback))

        # Stage 2: Self-correction attempt, retry once for each failed tactic action
        if correction_states and self._budget.used < self._budget.max_nodes:
            second_round_actions = self.policy.sample(
                correction_states, "tactic", 1, prompts=correction_prompts
            )
            # Verify the corrected actions in parallel
            self.executor.execute(
                graph,
                correction_states,
                second_round_actions,
                "tactic",
                self._budget,
                prompts=correction_prompts,
            )

    def _run_skeleton_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        skel_prompts = [build_prompt(s, "skeleton") for s in frontier]
        skel_batches = self.policy.sample(frontier, "skeleton", self.K_skel, prompts=skel_prompts)
        self.executor.execute(
            graph, frontier, skel_batches, "skeleton", self._budget, prompts=skel_prompts
        )


# File: betazero/search/rollout/batch_executor.py
from __future__ import annotations

import concurrent.futures
import threading

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.policy.prompt import build_prompt
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator

from .execution_result import LeanExecutionResult
from .failure_handler import FailureHandler
from .utils import format_lean_feedback


class RolloutBudget:
    __slots__ = ("max_nodes", "used", "_lock")

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        with self._lock:
            if self.used >= self.max_nodes:
                return False
            self.used += 1
            return True


class BatchExecutor:
    """Parallel Lean execute + expand graph; tactic feedbacks align with action_batches[i][j]."""

    def __init__(
        self,
        lean: LeanEnv,
        failure_handler: FailureHandler,
        reward: RewardCalculator,
        max_workers: int | None = None,
    ):
        self.lean = lean
        self.failure = failure_handler
        self.reward = reward
        # Get max workers from the executor to synchronize, avoid context switching.
        ex = getattr(lean.scheduler, "executor", None)
        self._max_workers = max_workers if max_workers is not None else (
            getattr(ex, "_max_workers", 4) if ex is not None else 4
        )

    @staticmethod
    def safe_execute(lean: LeanEnv, state: ProofState, action_code: str) -> LeanExecutionResult:
        """Run Lean; never raises — transport/executor errors become `system_errors` on the result."""
        try:
            sc, vr, sg = lean.execute(state, action_code)
            return LeanExecutionResult.ok(sc, vr, sg)
        except Exception as e:
            try:
                sc = lean._build_cmd(state, action_code)
            except Exception:
                sc = ""
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}", sc)

    def execute(
        self,
        graph: ANDORGraph,
        states: list[ProofState],
        action_batches: list[list[str]],
        action_type: str,
        budget: RolloutBudget,
        prompts: list[str] | None = None,
    ) -> list[list[tuple[str, str, str] | None]]:
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]

        tasks: list[tuple[int, int, ProofState, str, str, concurrent.futures.Future]] = []
        feedbacks: list[list[tuple[str, str, str] | None]] = [
            [None] * len(actions) for actions in action_batches
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for i, (state, actions) in enumerate(zip(states, action_batches)):
                for j, raw_output in enumerate(actions):
                    if not budget.try_consume():
                        break
                    lean_code = get_lean_code(raw_output)
                    if not lean_code:
                        self.failure.handle_system_execute_failure(
                            graph,
                            state,
                            action_type,
                            raw_output,
                            LeanExecutionResult.from_transport_error("empty_lean_code"),
                            prompts[i],
                        )
                        continue
                    fut = pool.submit(BatchExecutor.safe_execute, self.lean, state, lean_code)
                    tasks.append((i, j, state, raw_output, lean_code, fut))
                if budget.used >= budget.max_nodes:
                    break

            for i, j, state, raw_output, lean_code, future in tasks:
                res: LeanExecutionResult = future.result()
                prompt = prompts[i]
                if res.has_system_failure:
                    self.failure.handle_system_execute_failure(
                        graph, state, action_type, raw_output, res, prompt
                    )
                    continue
                state_code, state_vr, subgoals = res.state_code, res.verify, list(res.subgoals)
                r_env = self.reward.r_env(state_code, state_code, state_vr)
                if state_vr.get("complete"):
                    if action_type not in ("tactic", "skeleton"):
                        raise ValueError(f"Invalid action type: {action_type}")
                    act = Action(action_type, raw_output, (), prompt=prompt)
                    graph.expand(
                        state,
                        act,
                        r_env=r_env,
                        tactic_status="SOLVED" if action_type == "tactic" else None,
                    )
                    if action_type == "skeleton":
                        graph.set_skeleton_override(act, True)
                elif action_type == "tactic":
                    sorr_body = self.failure.handle_failed_tactic(
                        graph, state, raw_output, state_code, state_vr, prompt
                    )
                    feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), sorr_body)
                elif action_type == "skeleton":
                    if state_vr.get("pass"):
                        graph.expand(
                            state,
                            Action("skeleton", raw_output, tuple(subgoals), prompt=prompt),
                            r_env=r_env,
                        )
                    else:
                        self.failure.handle_failed_skeleton(
                            graph, state, raw_output, state_code, state_vr, prompt
                        )
                else:
                    raise ValueError(f"Invalid action type: {action_type}")

        return feedbacks


# File: betazero/search/rollout/__init__.py
"""Rollout pipeline for sampling, execution, repair, and graph expansion."""

from .execution_result import LeanExecutionResult
from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler
from .levelwise_rollout import LevelwiseRollout, SamplePolicy
from betazero.search.reward import DependencyRewardAssigner

__all__ = [
    "BatchExecutor",
    "DependencyRewardAssigner",
    "FailureHandler",
    "LeanExecutionResult",
    "LevelwiseRollout",
    "RolloutBudget",
    "SamplePolicy",
]


# File: betazero/search/rollout/utils.py
"""Lean example-wrapper helpers and verify message formatting."""


def extract_action_body(state_code: str) -> str:
    """Strip the `example ... := by` wrapper from compiled state code."""
    if ":= by\n" in state_code:
        body = state_code.split(":= by\n", 1)[1]
        return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
    return state_code


def format_lean_feedback(vr: dict) -> str:
    lines = [e.get("data", "") for e in vr.get("errors", [])[:12] if e.get("data", "")]
    if vr.get("system_errors"):
        lines.append(str(vr["system_errors"])[:800])
    return "\n".join(lines)


# File: betazero/search/rollout/failure_handler.py
from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .execution_result import LeanExecutionResult
from .utils import extract_action_body


class FailureHandler:
    """Sorrify failed tactics/skeletons and register penalized / patched graph edges."""

    def __init__(self, lean: LeanEnv, sorrifier: Sorrifier, reward: RewardCalculator):
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward

    def handle_system_execute_failure(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_kind: str,
        action_code: str,
        result: LeanExecutionResult,
        prompt: str = "",
    ) -> None:
        """Timeout / crash / transport errors: penalize graph edge; do not run sorrifier."""
        sc = result.state_code
        if not sc:
            try:
                sc = self.lean._build_cmd(state, action_code)
            except Exception:
                sc = action_code
        vr = result.verify
        r = self.reward.r_env(sc, sc, vr)
        graph.expand(
            state,
            Action(action_kind, action_code, (), prompt=prompt),
            r_env=r,
            tactic_status="FAILED" if action_kind == "tactic" else None,
        )

    def handle_failed_tactic(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
    ) -> str:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        graph.expand(
            state, Action("tactic", action_code, (), prompt=prompt), r_env=r_fail, tactic_status="FAILED"
        )
        return extract_action_body(patched)

    def handle_failed_skeleton(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
    ) -> None:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        patched_action_code = extract_action_body(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        graph.expand(state, Action("skeleton", action_code, (), prompt=prompt), r_env=r_fail)
        r_patch = self.reward.r_env(patched, patched, patched_vr)
        new_subgoals = [
            self.lean._parse_proof_state(s.get("goal", ""), header=state.header)
            for s in patched_vr.get("sorries", [])
        ]
        graph.expand(
            state,
            Action("skeleton", patched_action_code, tuple(new_subgoals), prompt=prompt),
            r_env=r_patch,
        )


# File: betazero/search/graph/__init__.py
"""Graph primitives for proof-state expansion and value backup."""

from .and_or_graph import ANDORGraph

__all__ = ["ANDORGraph"]


# File: betazero/search/graph/and_or_graph.py
from __future__ import annotations

import threading
from typing import Any, Literal

from betazero.core.nodes import Action, NodeStatus, ProofState
from betazero.policy.output_parser import get_lean_code
from betazero.search.sorrifier.stitcher import ProofStitcher


class ANDORGraph:
    """Thread-safe AND/OR proof graph with solved-state checks and return backup."""

    def __init__(self, root: ProofState):
        self._lock = threading.RLock()
        self._actions: dict[ProofState, list[Action]] = {root: []}
        self._parent: dict[Action, ProofState] = {}
        self._r_env: dict[Action, float] = {}
        self._r_dep: dict[Action, float] = {}
        self._tactic_status: dict[Action, Literal["SOLVED", "FAILED"]] = {}
        self._depth: dict[ProofState, int] = {root: 0}
        self._solved_cache: dict[Any, bool] = {}
        self._skeleton_override: dict[Action, bool] = {} 

    def expand(
        self,
        state: ProofState,
        action: Action,
        r_env: float = 0.0,
        r_dep: float = 0.0,
        tactic_status: Literal["SOLVED", "FAILED"] | None = None,
    ) -> None:
        with self._lock:
            if action in self._parent:
                return
            self._solved_cache.clear()
            self._actions.setdefault(state, []).append(action)
            self._parent[action] = state
            self._r_env[action] = r_env
            self._r_dep[action] = r_dep
            if tactic_status is not None and action.action_type == "tactic":
                self._tactic_status[action] = tactic_status
            for child in action.children:
                self._actions.setdefault(child, [])
                if child not in self._depth:
                    self._depth[child] = self._depth[state] + 1

    def _node_solved(
        self, node: ProofState | Action, visiting: set, memo: dict[Any, bool]
    ) -> bool:
        if node in visiting:
            return False
        if node in memo:
            return memo[node]
        visiting.add(node)
        try:
            if isinstance(node, ProofState):
                res = any(self._node_solved(a, visiting, memo) for a in self._actions.get(node, []))
            elif node.action_type == "tactic":
                res = self._tactic_status.get(node) == "SOLVED"
            else:
                if node in self._skeleton_override:
                    res = self._skeleton_override[node]
                else:
                    res = bool(node.children) and all(
                        self._node_solved(c, visiting, memo) for c in node.children
                    )
            memo[node] = res
            return res
        finally:
            visiting.remove(node)

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        with self._lock:
            if visiting is None:
                visiting = set()
            return self._node_solved(node, visiting, self._solved_cache)

    def status(self, node: ProofState | Action) -> NodeStatus:
        with self._lock:
            if isinstance(node, ProofState):
                return "SOLVED" if self.is_solved(node) else "OPEN"
            if node.action_type == "tactic":
                t = self._tactic_status.get(node)
                if t == "SOLVED":
                    return "SOLVED"
                if t == "FAILED":
                    return "FAILED"
                return "OPEN"
            if self.is_solved(node):
                return "SOLVED"
            if not node.children:
                return "FAILED"
            return "OPEN"

    def unsolved_states(self) -> list[ProofState]:
        with self._lock:
            keys = list(self._actions.keys())
        return [s for s in keys if not self.is_solved(s)]

    def get_actions(self, state: ProofState) -> list[Action]:
        with self._lock:
            return list(self._actions.get(state, []))

    def get_r_env(self, action: Action) -> float:
        with self._lock:
            return self._r_env.get(action, 0.0)

    def get_parent(self, action: Action, default: ProofState | None = None) -> ProofState | None:
        with self._lock:
            return self._parent.get(action, default)

    def parent_items(self) -> list[tuple[Action, ProofState]]:
        with self._lock:
            return list(self._parent.items())

    def set_r_dep(self, action: Action, r_dep: float) -> None:
        with self._lock:
            self._r_dep[action] = r_dep

    def set_skeleton_override(self, action: Action, is_solved: bool):
        with self._lock:
            self._skeleton_override[action] = is_solved
            self._solved_cache.clear() # Nhớ xóa cache để graph tính lại từ đầu

    def get_depth(self, state: ProofState) -> int:
        with self._lock:
            return self._depth.get(state, -1)

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        with self._lock:
            q_cache: dict[Action, float] = {}
            v_cache: dict[ProofState, float] = {}
            visiting_v: set[ProofState] = set()
            solve_memo: dict[Any, bool] = {}

            def V(state: ProofState) -> float:
                if state in v_cache:
                    return v_cache[state]
                if state in visiting_v:
                    return 0.0
                visiting_v.add(state)
                val = max((Q(a) for a in self._actions.get(state, [])), default=0.0)
                visiting_v.remove(state)
                v_cache[state] = val
                return val

            def Q(action: Action) -> float:
                if action in q_cache:
                    return q_cache[action]
                r_e = self._r_env.get(action, 0.0)
                solved = self._node_solved(action, set(), solve_memo)
                if action.action_type == "tactic":
                    val = r_e + W_solve * float(solved)
                else:
                    r_d = self._r_dep.get(action, 0.0)
                    future = gamma * min((V(c) for c in action.children), default=0.0)
                    val = r_e + float(solved) * (r_d + future)
                q_cache[action] = val
                return val

            for action in self._parent:
                Q(action)
            for state in self._actions:
                V(state)
            return q_cache

    def get_successful_action(self, state: ProofState) -> Action | None:
        """Retrieve the action that successfully solved this state."""
        with self._lock:
            for action in self.get_actions(state):
                if self.status(action) == "SOLVED":
                    return action
        return None

    def extract_proof_code(self, state: ProofState) -> str | None:
        """Recursively extract and stitch the successful proof code for a state."""
        
        action = self.get_successful_action(state)
        if not action:
            return None
            
        parsed_code = get_lean_code(action.content)
        
        if action.action_type == "tactic":
            return parsed_code
            
        # Skeleton: recurse down to children
        child_proofs = [self.extract_proof_code(child) for child in action.children]
        return ProofStitcher.stitch(parsed_code, child_proofs)

# File: betazero/search/reward/reward_assigner.py
"""Assigns structural dependencies reward (r_dep) to skeleton actions."""

from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph
from betazero.search.sorrifier.stitcher import ProofStitcher
from .calculator import RewardCalculator


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis to assign dependency rewards."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def assign(self, graph: ANDORGraph) -> None:
        """Bottom-up dependency reward assignment using Expr Trees."""
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue
            
            # 1. Collect child proofs
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            
            # 2. Stitch code
            skeleton_code = get_lean_code(action.content)
            stitched_code = ProofStitcher.stitch(skeleton_code, child_proofs)
            full_compilable_code = self.lean._build_cmd(parent_state, stitched_code)
            
            # 3. Analyze through Kernel Expr Tree
            dep_analysis = self.lean.analyze_dependencies(full_compilable_code)
            
            # 4. Map outputs to Calculator format and assign
            mapped_analysis = {
                "core": dep_analysis.get("core_solved", []) + dep_analysis.get("core_failed", []),
                "benign": dep_analysis.get("benign", []),
                "malignant": dep_analysis.get("malignant", [])
            }
            
            r_dep_score = self.reward.r_dep(mapped_analysis)
            
            # Fatal penalty for missing core subgoals
            if dep_analysis.get("core_failed"):
                r_dep_score = -1.0 
                
            graph.set_r_dep(action, r_dep_score)

# File: betazero/search/reward/calculator.py
from betazero.env.ast_parser import get_lean_ast
from betazero.search.graph import ANDORGraph
from betazero.core import Action


class RewardCalculator:
    """Computes environment and dependency rewards, then backs them up through the graph."""

    def __init__(self, W_c: float = 1.0, W_b: float = 0.0, W_m: float = -1.0,
                 W_solve: float = 1.0, gamma: float = 1.0):
        assert W_c > W_b and W_b <= 0 and W_m < W_b, "Required: W_m < W_b <= 0 < W_c"
        self.W_c, self.W_b, self.W_m = W_c, W_b, W_m
        self.W_solve = W_solve
        self.gamma = gamma

    @staticmethod
    def _categorize_nodes(ast_nodes: list, dead_lines: set = None) -> tuple[int, int, int, int]:
        """Classify tactic AST nodes. Returns (total, sorries, junk, dead)."""
        total = sorries = junk = dead = 0
        dead_lines = dead_lines or set()
        for n in ast_nodes:
            kind = n.get("kind", "")
            if not kind or not kind.startswith("Lean.Parser.Tactic."):
                continue
            low = kind.lower()
            if "seq" in low:
                continue
            total += 1
            if "sorry" in low:
                sorries += 1
            elif "skip" in low or "done" in low:
                junk += 1
            elif n.get("pos", {}).get("line") in dead_lines:
                dead += 1
        return total, sorries, junk, dead

    def r_env(self, original_code: str, patched_code: str, verify_result: dict) -> float:
        """Ratio of surviving valid semantic nodes after patching (Section 6.1)."""
        ast_orig = get_lean_ast(original_code)
        tot_orig, sorries_orig, _, _ = self._categorize_nodes(ast_orig)
        if tot_orig == 0:
            return 0.0
        dead_lines = {
            w["pos"]["line"]
            for w in verify_result.get("warnings", [])
            if "unused" in w.get("data", "").lower() or "does nothing" in w.get("data", "").lower()
        }
        ast_patched = get_lean_ast(patched_code)
        tot_patch, sorries_patch, junk_patch, dead_patch = self._categorize_nodes(ast_patched, dead_lines)
        new_sorries = max(0, sorries_patch - sorries_orig)  # sorries added by patcher, not model
        t_valid = max(0, tot_patch - junk_patch - new_sorries - dead_patch)
        return min(1.0, t_valid / tot_orig)

    def r_dep(self, dep_graph: dict) -> float:
        """Weighted dependency reward (Section 6.2)."""
        n_c = len(dep_graph.get("core", []))
        n_b = len(dep_graph.get("benign", []))
        n_m = len(dep_graph.get("malignant", []))
        h_total = n_c + n_b + n_m
        if h_total == 0:
            return 0.0
        return (self.W_c * n_c + self.W_b * n_b + self.W_m * n_m) / h_total

    def compute_returns(self, graph: ANDORGraph) -> dict[Action, float]:
        return graph.backup(gamma=self.gamma, W_solve=self.W_solve)


# File: betazero/search/reward/__init__.py
"""Reward computation for local execution quality and dependency structure."""

from .calculator import RewardCalculator
from .reward_assigner import DependencyRewardAssigner

__all__ = ["DependencyRewardAssigner", "RewardCalculator"]


# File: betazero/search/trainer/grpo_trainer.py
from __future__ import annotations
import random
from collections import defaultdict

import torch

from betazero.core import ProofState, Action
from betazero.policy import TrainablePolicy

_EMPTY = {"loss": 0.0, "kl": 0.0, "n_samples": 0, "n_groups": 0,
          "r_env_mean": 0.0, "Q_mean": 0.0, "solve_rate": 0.0}


class GRPOTrainer:
    """Applies GRPO updates to a policy from rollout samples collected during search."""

    def __init__(self, lr: float = 1e-5, eps_clip: float = 0.2, beta_kl: float = 0.01,
                 grpo_epochs: int = 1, mini_batch_size: int = 8, accumulation_steps: int = 1):
        self.lr                 = lr
        self.eps_clip           = eps_clip
        self.beta_kl            = beta_kl
        self.grpo_epochs        = grpo_epochs
        self.mini_batch_size    = mini_batch_size
        self.accumulation_steps = max(1, accumulation_steps)

    def update(self, policy: TrainablePolicy,
               samples: list[tuple[ProofState, Action, float, float]],
               aux_samples: list[tuple[ProofState, Action, float, float]] | None = None) -> dict:
        """GRPO update on pre-collected rollout samples. Returns metrics dict."""
        if not samples:
            return _EMPTY

        # Gom nhóm toàn bộ sample theo prompt
        raw_groups: dict[str, list[int]] = defaultdict(list)
        for i, (_, action, _, _) in enumerate(samples):
            raw_groups[action.prompt].append(i)

        adv  = [0.0] * len(samples)
        keep = [False] * len(samples)
        
        # Chỉ lấy các nhóm đủ điều kiện làm GRPO (size >= 2 và std > 0)
        valid_prompt_groups: dict[str, list[int]] = {}
        
        for p_text, idxs in raw_groups.items():
            returns = [
                samples[i][2] if samples[i][1].action_type == "tactic" else samples[i][3]
                for i in idxs
            ]
            mean_r  = sum(returns) / len(returns)
            std_r   = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
            
            if std_r < 1e-8 or len(idxs) < 2:
                continue
                
            group_keep_idxs = []
            for i, r in zip(idxs, returns):
                adv[i] = (r - mean_r) / std_r
                keep[i] = True
                group_keep_idxs.append(i)
                
            valid_prompt_groups[p_text] = group_keep_idxs

        train_idx = [i for i in range(len(samples)) if keep[i]]
        if not train_idx:
            return _EMPTY

        # Bóc tách dữ liệu sạch để train
        # Cần map index cũ -> index mới trong danh sách train_idx
        old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate(train_idx)}
        train_groups = {
            p_text: [old_to_new_idx[i] for i in idxs]
            for p_text, idxs in valid_prompt_groups.items()
        }

        states  = [samples[i][0] for i in train_idx]
        actions = [samples[i][1].content for i in train_idx]
        prompts = [samples[i][1].prompt for i in train_idx]
        device  = next(policy.parameters()).device
        adv_t   = torch.tensor([adv[i] for i in train_idx], dtype=torch.float32, device=device)

        # Tính Reference Log-probs
        with torch.no_grad():
            ref_lp = self._batch_lp(policy, states, actions, prompts, train_groups, disable_adapter=True)

        optimizer  = torch.optim.Adam(policy.parameters(), lr=self.lr)
        total_loss = total_kl = 0.0
        n_steps    = 0
        acc        = self.accumulation_steps

        for _ in range(self.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            
            # Tạo mini-batches sao cho bảo toàn cụm prompt
            mb_list = self._minibatches(train_groups)
            n_mb = len(mb_list)
            
            for mb_idx, mb in enumerate(mb_list):
                mb_s = [states[j] for j in mb]
                mb_a = [actions[j] for j in mb]
                mb_p = [prompts[j] for j in mb]
                mb_adv = adv_t[mb]
                mb_ref = ref_lp[mb]

                mb_theta = policy.log_probs(mb_s, mb_a, mb_p)
                ratio    = torch.exp(mb_theta - mb_ref)
                surr     = torch.min(ratio * mb_adv,
                                     torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv)
                kl   = ratio - (mb_theta - mb_ref) - 1.0
                loss = -(surr - self.beta_kl * kl).mean() / acc

                loss.backward()
                accum_counter += 1
                is_last = mb_idx + 1 == n_mb
                
                if accum_counter >= acc or is_last:
                    if is_last and accum_counter < acc:
                        scale = acc / accum_counter
                        for p in policy.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                total_loss += loss.item() * acc
                total_kl   += kl.mean().item()
                n_steps    += 1

        r_envs   = [s[2] for s in samples]
        qs       = [s[3] for s in samples]
        return {
            "loss":       total_loss / n_steps,
            "kl":         total_kl   / n_steps,
            "n_samples":  len(train_idx),
            "n_groups":   len(valid_prompt_groups),
            "r_env_mean": sum(r_envs) / len(r_envs),
            "Q_mean":     sum(qs)     / len(qs),
            "solve_rate": sum(1 for s in samples if s[3] > 0) / len(samples),
        }

    def _minibatches(self, groups_dict: dict[str, list[int]]):
        """
        Chia mini-batch sao cho các action chung prompt được đóng gói cùng nhau.
        Tối đa hóa hiệu năng của Shared Prefill trong TrainablePolicy.
        """
        all_mbs = []
        prompts = list(groups_dict.keys())
        random.shuffle(prompts)
        
        for p in prompts:
            idxs = groups_dict[p].copy()
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.mini_batch_size):
                all_mbs.append(idxs[i:i + self.mini_batch_size])
                
        random.shuffle(all_mbs)
        return all_mbs

    def _batch_lp(self, policy, states, actions, prompts, groups_dict,
                  disable_adapter: bool = False) -> torch.Tensor:
        """Dùng logic minibatches gom nhóm để tính reference log probs tốc độ cao."""
        results = torch.zeros(len(states), dtype=torch.float32, device=next(policy.parameters()).device)
        
        mb_list = self._minibatches(groups_dict)
        for mb in mb_list:
            s = [states[i] for i in mb]
            a = [actions[i] for i in mb]
            p = [prompts[i] for i in mb]
            
            mb_scores = policy.log_probs(s, a, p, disable_adapter=disable_adapter)
            for i, score in zip(mb, mb_scores):
                results[i] = score
                
        return results

# File: betazero/search/trainer/__init__.py
"""Policy optimization routines over collected search samples."""

from .grpo_trainer import GRPOTrainer

__all__ = ["GRPOTrainer"]


# File: betazero/env/expr_parser.py
"""Persistent Lean EXPR Tree daemon. Single shared process across all imports."""

import atexit
import json
import os
import signal
import subprocess
import tempfile
import threading

REPL_DIR = os.environ.get("LEAN_WORKSPACE", os.path.join(os.getcwd(), "repl/"))


class EXPRTreeDaemon:
    def __init__(self, repl_dir: str = REPL_DIR, max_requests: int = 500):
        self.repl_dir = repl_dir
        self.max_requests = max_requests
        self.request_count = 0
        self.lock = threading.Lock()
        self.proc = None
        self._start_process()
        atexit.register(self.close)

    def _kill_proc(self):
        proc_to_kill = self.proc
        self.proc = None
        if proc_to_kill:
            try:
                if proc_to_kill.poll() is None:
                    os.killpg(os.getpgid(proc_to_kill.pid), signal.SIGKILL)
                    proc_to_kill.wait(timeout=2)
            except Exception:
                pass

    def _start_process(self):
        self._kill_proc()
        self.request_count = 0
        self.proc = subprocess.Popen(
            ["lake", "exe", "dump_expr_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.repl_dir,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.repl_dir, delete=False, encoding="utf-8"
        ) as tf:
            tf.write('import Mathlib')
            tmp = tf.name
        try:
            print("[EXPR] Loading Mathlib environment...")
            self._get_expr_raw(tmp)
            print("[EXPR] Ready.")
        finally:
            os.remove(tmp)

    def _get_expr_raw(self, file_path: str) -> list:
        if not self.proc or not self.proc.stdin:
            return []
        self.proc.stdin.write(file_path + "\n")
        self.proc.stdin.flush()
        blocks = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "===EOF===":
                break
            if line.startswith("{"):
                try:
                    blocks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return blocks

    def get_expr_tree(self, file_path: str) -> list:
        with self.lock:
            self.request_count += 1
            if (
                self.proc is None
                or self.proc.poll() is not None
                or self.request_count >= self.max_requests
            ):
                if self.request_count >= self.max_requests:
                    print(
                        f"[EXPR] Reached {self.max_requests} requests. Respawning to flush RAM..."
                    )
                else:
                    print("[EXPR] Daemon crashed or missing. Restarting...")
                self._start_process()
            return self._get_expr_raw(file_path)

    def close(self):
        self._kill_proc()

_SHARED_TREE_DAEMON = EXPRTreeDaemon(REPL_DIR)

def get_lean_expr_tree(code: str) -> list:
    prefix = "import Mathlib\n"
    full_code = code if "import " in code else prefix + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=REPL_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        path = f.name
        
    try:
        return _SHARED_TREE_DAEMON.get_expr_tree(path)
    finally:
        os.remove(path)

# File: betazero/env/__init__.py
from .lean_verifier import Lean4ServerScheduler

__all__ = [
    "Lean4ServerScheduler",
]

# File: betazero/env/ast_parser.py
"""Persistent Lean AST daemon. Single shared process across all imports."""

import atexit
import json
import os
import signal
import subprocess
import tempfile
import threading

REPL_DIR = os.environ.get("LEAN_WORKSPACE", os.path.join(os.getcwd(), "repl/"))


class ASTDaemon:
    def __init__(self, repl_dir: str = REPL_DIR, max_requests: int = 500):
        self.repl_dir = repl_dir
        self.max_requests = max_requests
        self.lock = threading.Lock()
        self.proc = None
        self._start_process()
        atexit.register(self.close)

    def _kill_proc(self):
        proc_to_kill = self.proc
        self.proc = None
        if proc_to_kill:
            try:
                if proc_to_kill.poll() is None:
                    os.killpg(os.getpgid(proc_to_kill.pid), signal.SIGKILL)
                    proc_to_kill.wait(timeout=2)
            except Exception:
                pass

    def _start_process(self):
        self._kill_proc()
        self.request_count = 0
        
        self.proc = subprocess.Popen(
            ["lake", "exe", "dump_ast_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.repl_dir,
            bufsize=1,
            preexec_fn=os.setsid  
        )
        
        # Warmup: cache Mathlib environment so the first real call is fast
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=self.repl_dir, delete=False, encoding="utf-8"
        ) as tf:
            tf.write('import Mathlib')
            tmp = tf.name
        try:
            print("[AST] Loading Mathlib environment...")
            self._get_ast_raw(tmp)
            print("[AST] Ready.")
        finally:
            os.remove(tmp)

    def _get_ast_raw(self, file_path: str) -> list:
        self.proc.stdin.write(file_path + "\n")
        self.proc.stdin.flush()
        blocks = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "===EOF===":
                break
            if line.startswith("{"):
                try:
                    blocks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return blocks

    def get_ast(self, file_path: str) -> list:
        with self.lock:
            self.request_count += 1
            
            if self.proc is None or self.proc.poll() is not None or self.request_count >= self.max_requests:
                if self.request_count >= self.max_requests:
                    print(f"\n[AST Server] Reached {self.max_requests} requests. Respawning to flush RAM leak =))...")
                self._start_process()

            try:
                return self._get_ast_raw(file_path)
            except Exception as e:
                print(f"[AST Server] Error during AST dump: {e}. Respawning...")
                self._start_process()  
                return []

    def close(self):
        self._kill_proc()


_SHARED_DAEMON = ASTDaemon()


def get_lean_ast(code: str) -> list:
    """Return AST block dicts for a Lean code string. Thread-safe with auto-import injection and offset correction."""
    
    prefix = "import Mathlib\n"
    has_import = "import " in code
    
    if not has_import:
        full_code = prefix + code
        offset_bytes = len(prefix.encode('utf-8')) # Chính xác là 15 bytes
    else:
        full_code = code
        offset_bytes = 0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=REPL_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        path = f.name
        
    try:
        blocks = _SHARED_DAEMON.get_ast(path)
        
        if offset_bytes > 0:
            valid_blocks = []
            for b in blocks:
                b["start_byte"] -= offset_bytes
                b["end_byte"] -= offset_bytes
                
                # Chỉ lấy những node thuộc về code thật (>= 0), vứt bỏ cái node của dòng import giả mạo
                if b["start_byte"] >= 0:
                    valid_blocks.append(b)
            return valid_blocks
            
        return blocks
    finally:
        os.remove(path)


# File: betazero/env/lean_env.py
import re

from betazero.core import ProofState
from betazero.env.ast_parser import get_lean_ast
from betazero.env import Lean4ServerScheduler
from betazero.env.expr_parser import get_lean_expr_tree


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        candidate_code = self._build_cmd(state, code)
        vr = self.scheduler.verify(candidate_code)
        
        subgoals = []
        if vr.get("pass"):
            for s in vr.get("sorries", []):
                ps = self._parse_proof_state(s.get("goal", ""), header=state.header)
                if ps.goal not in ["SOLVED_OR_EMPTY", "ELABORATION_FAULT"]:
                    subgoals.append(ps)

        return candidate_code, vr, subgoals

    def get_ast(self, code: str) -> list:
        return get_lean_ast(code)

    def analyze_dependencies(self, proof_code: str) -> dict:
        """
        Classify subgoals using Lean 4 Expr Tree deep analysis.
        Returns classifications for: core_solved, core_failed, malignant, benign.
        """
        from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
        ast_expr_list = get_lean_expr_tree(proof_code)
        
        empty_classification = {
            "core_solved": [], "core_failed": [], "malignant": [], "benign": []
        }
        
        if not ast_expr_list:
            return empty_classification
            
        root_expr = ast_expr_list[-1].get("expr_tree", {})
        classification = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(root_expr)
        
        return classification

    @staticmethod
    def _parse_proof_state(goal_str: str, header: str = "") -> ProofState:
        """Parse Lean Infoview goal string into a ProofState (with inherited header)."""
        s = goal_str.strip()
        
        if not s or "Goals accomplished" in s or "no goals" in s:
            return ProofState(context="", goal="SOLVED_OR_EMPTY", header=header)

        parts = s.split("⊢")
        
        if len(parts) > 1:
            ctx_raw = parts[0].strip()
            main_goal_raw = parts[1].strip()
            
            goal_lines = []
            for line in main_goal_raw.splitlines():
                if line.startswith("case ") or "Goals accomplished" in line:
                    break
                goal_lines.append(line)
            goal = "\n".join(goal_lines).strip()
        else:
            ctx_raw, goal = "", s.strip()

        if goal.lower() == "sorry":
            goal = "ELABORATION_FAULT"

        valid_ctx_lines = []
        if ctx_raw:
            for line in ctx_raw.splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("case"):
                    valid_ctx_lines.append(line)
                    
        ctx = "\n".join(valid_ctx_lines)

        return ProofState(context=ctx, goal=goal, header=header)

    @staticmethod
    def _sanitize_header(header: str) -> str:
        """Remove redundant Mathlib imports and normalize maxHeartbeats for Lean execution."""
        if not header:
            return ""

        out: list[str] = []
        for line in header.splitlines():
            if re.match(r'^\s*import\s+Mathlib', line):
                continue
            if re.match(r'^\s*set_option\s+maxHeartbeats\s+0\s*$', line):
                out.append("set_option maxHeartbeats 100000")
                continue
            out.append(line)
        return "\n".join(out).strip()

    @staticmethod
    def _build_cmd(state: ProofState, code: str) -> str:
        """Wrap state header, context and goal into a compilable theorem block."""
        params = [
            f"({line.strip()})"
            for line in state.context.splitlines()
            if line.strip() and ":" in line and not line.strip().startswith("case ")
        ] if state.context else []

        param_str = (" ".join(params) + " ") if params else ""
        indented = "\n".join(f"  {l}" for l in code.strip().splitlines())
        prefix_header = LeanEnv._sanitize_header(state.header) if state.header else ""
        prefix = (prefix_header + "\n\n") if prefix_header else ""
        # Use a stable, named declaration so downstream tooling (e.g. expr dump) can reliably find it.
        return f"{prefix}theorem __bz_tmp {param_str}: {state.goal} := by\n{indented}"

# File: betazero/env/lean_verifier.py
import os
import time
import json
import subprocess
import shutil
import uuid
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")


import os
import signal # Thêm cái này
import time
import json
import subprocess
import shutil
import uuid
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

DEFAULT_LAKE_PATH = shutil.which("lake") or "lake"
DEFAULT_LEAN_WORKSPACE = os.path.join(os.getcwd(), "repl/")

class PersistentLeanWorker:
    """A persistent Lean REPL process that caches Mathlib in its base environment."""
    
    def __init__(self, workspace=DEFAULT_LEAN_WORKSPACE, timeout=20, max_requests=500): 
        self.workspace = workspace
        self.timeout = timeout
        self.max_requests = max_requests 
        self.request_count = 0           
        self.proc = None
        self.base_env = None
        self.lock = threading.Lock()
        self._is_timed_out = False # Cờ báo hiệu bom nổ
        self._start_repl()

    def _kill_proc(self):
        # Lấy proc ra và gán None ngay lập tức để tránh Race Condition với Thread bom
        proc_to_kill = self.proc
        self.proc = None
        if proc_to_kill:
            try:
                if proc_to_kill.poll() is None:
                    os.killpg(os.getpgid(proc_to_kill.pid), signal.SIGKILL)
                    proc_to_kill.wait(timeout=2)
            except Exception as e:
                pass # Nuốt lỗi vì process có thể đã chết

    def _timeout_handler(self):
        """Hàm kích nổ: Gây án mạng khi hết giờ."""
        self._is_timed_out = True
        self._kill_proc()

    def _read_json_response(self):
        buffer = ""
        while True:
            # Nếu _kill_proc() được gọi từ Timer, proc sẽ bị None hoặc pipe bị đứt
            if not self.proc or not self.proc.stdout:
                raise RuntimeError("Process killed by timeout or died.")
                
            line = self.proc.stdout.readline()
            
            if not line:
                if buffer:
                    print(f"\n[CRITICAL FATAL] Lean REPL đột tử khi đang in JSON! Code dở dang:\n{buffer}")
                raise RuntimeError("REPL closed unexpectedly (EOF). Lean Process died.")
            
            if not buffer and not line.strip().startswith("{"):
                print(f"[REPL GARBAGE] {line.strip()}")
                continue
            
            buffer += line
            
            try:
                parsed_json = json.loads(buffer)
                return parsed_json
            except json.JSONDecodeError:
                continue

    def _start_repl(self):
        self._kill_proc()
        self.request_count = 0
        self._is_timed_out = False
        
        self.proc = subprocess.Popen(
            [DEFAULT_LAKE_PATH, "exe", "repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.workspace,
            bufsize=1,
            preexec_fn=os.setsid
        )
        warmup_cmd = json.dumps({"cmd": "import Mathlib"})
        
        self.proc.stdin.write(warmup_cmd + "\n\n")
        self.proc.stdin.flush()
        
        try:
            res_json = self._read_json_response()
            self.base_env = res_json.get("env")
            print(f"[Worker PID {self.proc.pid}] Mathlib warmed up. Base env: {self.base_env}")
        except Exception as e:
            print(f"[Worker PID {self.proc.pid if self.proc else 'N/A'}] Warmup failed: {e}")
            self.base_env = None

    def verify(self, code: str) -> dict:
        start_time = time.time()
        
        with self.lock:
            self.request_count += 1 
            
            if self.proc is None or self.proc.poll() is not None or self.request_count >= self.max_requests:
                if self.request_count >= self.max_requests:
                    print(f"Worker reached {self.max_requests} requests. Respawning to flush RAM leak...")
                else:
                    print("Worker died or missing. Respawning...")
                self._start_repl()

            payload = {"cmd": code}
            if self.base_env is not None:
                payload["env"] = self.base_env

            message_str = json.dumps(payload, ensure_ascii=False)
            
            # GÀI BOM HẸN GIỜ!
            self._is_timed_out = False
            timer = threading.Timer(self.timeout, self._timeout_handler)
            timer.start()
            
            try:
                self.proc.stdin.write(message_str + "\n\n")
                self.proc.stdin.flush()
                
                result_json = self._read_json_response()
                
                messages = result_json.get("messages", [])
                errors   = [m for m in messages if m.get("severity") == "error"]
                warnings = [m for m in messages if m.get("severity") == "warning"]
                sorries  = result_json.get("sorries", [])
                
                is_pass  = len(errors) == 0
                has_sorry_warning = any(
                    "declaration uses 'sorry'" in w.get("data", "") or "failed" in w.get("data", "")
                    for w in warnings
                )
                
                result = {
                    "pass": is_pass,
                    "complete": is_pass and not sorries and not has_sorry_warning,
                    "errors": errors,
                    "warnings": warnings,
                    "infos": [m for m in messages if m.get("severity") == "info"],
                    "sorries": sorries,
                    "system_errors": "",
                    "verified_code": code,
                }
            except Exception as e:
                # Kiểm tra xem có phải do bom nổ không
                if self._is_timed_out:
                    sys_err = f"Lean verification TIMED OUT after {self.timeout} seconds!"
                else:
                    sys_err = str(e)
                    
                result = {"pass": False, "complete": False, "system_errors": sys_err, "errors": []}
                print(f"[Worker] Failed/Timeout: {sys_err}. Respawning...")
                self._start_repl() # Thay máu công nhân mới
            finally:
                timer.cancel() # Gỡ bom nếu Lean chạy xong sớm!

        result["verify_time"] = time.time() - start_time
        return result

    def close(self):
        self._kill_proc()


class Lean4ServerScheduler:
    """Thread pool managing persistent Stateful Lean Workers."""

    def __init__(self, max_concurrent_requests=1, timeout=60, name="verifier", **kwargs):
        self.timeout = timeout
        self.worker_queue = Queue()
        self.workers = []
        
        print(f"[{name}] Booting {max_concurrent_requests} persistent Lean worker(s)... (This takes ~5 seconds)")
        for _ in range(max_concurrent_requests):
            worker = PersistentLeanWorker(timeout=self.timeout)
            self.workers.append(worker)
            self.worker_queue.put(worker)
            
        # Dùng ThreadPool thay vì ProcessPool vì các Worker object có chứa Popen subprocess
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.futures: dict = {}
        print(f"[{name}] Scheduler Ready.")

    def _verify_task(self, code: str):
        worker = self.worker_queue.get()
        try:
            res = worker.verify(code)
            return res
        finally:
            self.worker_queue.put(worker)

    def submit_all_request(self, tasks) -> list[str]:
        request_ids = []
        for task in tasks:
            req_id = str(uuid.uuid4())
            self.futures[req_id] = self.executor.submit(
                self._verify_task, task.get("code", "")
            )
            request_ids.append(req_id)
        return request_ids

    def get_all_request_outputs(self, request_ids) -> list[dict]:
        return [self.futures.pop(req_id).result() for req_id in request_ids if req_id in self.futures]

    def verify(self, code: str) -> dict:
        req_id = self.submit_all_request([{"code": code}])[0]
        return self.get_all_request_outputs([req_id])[0]

    def close(self):
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            worker.close()
        print("Lean4ServerScheduler closed and all REPLs terminated.")

