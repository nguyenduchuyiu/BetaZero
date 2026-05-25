import os
import gc
import torch
from collections import defaultdict
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from gammazero.core import ProofState
from gammazero.utils import Config


class TrainablePolicy:
    """PEFT model used for log-prob computation and gradient updates.

    Adapters are explicitly loaded/unloaded to free VRAM between phases.
    """

    def __init__(self, cfg: Config, adapter_path: str | None = None):
        self.device = cfg.device
        self.logprob_chunk_size = max(1, cfg.logprob_chunk_size)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, 
            quantization_config=bnb_config, 
            device_map=cfg.device
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()

        base_tactic = os.path.abspath(cfg.base_lora_tactic)
        base_skeleton = os.path.abspath(cfg.base_lora_skeleton)

        # Load reference adapters.
        self.model = PeftModel.from_pretrained(base, base_tactic, adapter_name="ref_tactic")
        if os.path.exists(base_skeleton):
            self.model.load_adapter(base_skeleton, adapter_name="ref_skeleton")

        # Load or initialize the active RL adapters.
        if adapter_path and os.path.exists(adapter_path):
            tactic_path = os.path.join(adapter_path, "tactic")
            skeleton_path = os.path.join(adapter_path, "skeleton")
            if os.path.exists(tactic_path):
                self.model.load_adapter(tactic_path, adapter_name="active_tactic", is_trainable=True)
            if os.path.exists(skeleton_path):
                self.model.load_adapter(skeleton_path, adapter_name="active_skeleton", is_trainable=True)
        else:
            # First iteration: clone the SFT base into trainable RL adapters.
            self.model.load_adapter(base_tactic, adapter_name="active_tactic", is_trainable=True)
            if os.path.exists(base_skeleton):
                self.model.load_adapter(base_skeleton, adapter_name="active_skeleton", is_trainable=True)
                
        self.model.print_trainable_parameters()

    def parameters(self):
        return self.model.parameters()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        tactic_path = os.path.join(path, "tactic")
        skeleton_path = os.path.join(path, "skeleton")
        
        self.model.set_adapter("active_tactic")
        self.model.save_pretrained(tactic_path)
        
        try:
            self.model.set_adapter("active_skeleton")
            self.model.save_pretrained(skeleton_path)
        except ValueError:
            pass

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def _score_chunk(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        chosen = torch.gather(logits, dim=2, index=targets.unsqueeze(-1)).squeeze(-1).float()
        norm = torch.logsumexp(logits.float(), dim=-1)
        return chosen - norm

    def log_probs(self, states: list[ProofState], actions: list[str],
                  prompts: list[str], action_types: list[str], disable_adapter: bool = False) -> torch.Tensor:
        """Compute log-probs by concatenating prompt + action directly.

        Skips a shared prefill and relies on FlashAttention-2, which keeps
        PyTorch's allocation/backprop semantics correct.
        """
        if not prompts:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        # Group inputs by (prompt, action_type) so we can switch adapters once per group.
        group_to_idxs = defaultdict(list)
        for i, (p, t) in enumerate(zip(prompts, action_types)):
            group_to_idxs[(p, t)].append(i)

        scores = torch.zeros(len(prompts), dtype=torch.float32, device=self.device)

        # Force right-padding so target indices line up; restore on exit.
        orig_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        try:
            for (p_text, a_type), idxs in group_to_idxs.items():
                group_actions = [actions[i] for i in idxs]

                # Switch to the appropriate adapter.
                target_adapter = f"ref_{a_type}" if disable_adapter else f"active_{a_type}"
                self.model.set_adapter(target_adapter)

                group_scores = self._score_direct(p_text, group_actions)

                for i, score in zip(idxs, group_scores):
                    scores[i] = score
        finally:
            self.tokenizer.padding_side = orig_padding

        return scores

    def _score_direct(self, prompt: str, actions: list[str]) -> list[torch.Tensor]:
        """Compute log-probs over action tokens by concatenating ids manually.

        Manual concatenation avoids BPE merges across the prompt/action boundary
        that would otherwise corrupt the alignment.
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        # Tokenize actions independently.
        encoded_actions = self.tokenizer(actions, add_special_tokens=False)

        # Concatenate ids and right-pad.
        batch_ids = []
        max_len = 0
        for act_ids in encoded_actions.input_ids:
            full_ids = prompt_ids + act_ids
            batch_ids.append(full_ids)
            if len(full_ids) > max_len:
                max_len = len(full_ids)

        B = len(actions)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

        for i, b_ids in enumerate(batch_ids):
            L = len(b_ids)
            input_ids[i, :L] = torch.tensor(b_ids, dtype=torch.long)
            attention_mask[i, :L] = 1

        target_mask = torch.zeros_like(attention_mask)

        # Mask only action tokens (logit at t predicts token t+1).
        start_idx = prompt_len - 1
        for i in range(B):
            end_idx = attention_mask[i].sum() - 1
            if end_idx > start_idx:
                target_mask[i, start_idx:end_idx] = 1

        action_logprobs = []
        # Chunk to avoid OOM.
        chunk_size = self.logprob_chunk_size
        for i in range(0, B, chunk_size):
            chunk_input_ids = input_ids[i:i+chunk_size]
            chunk_attention_mask = attention_mask[i:i+chunk_size]
            chunk_target_mask = target_mask[i:i+chunk_size]

            out = self.model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, use_cache=False)
            logits = out.logits[:, :-1, :]  # (B_chunk, T-1, V)
            teacher_targets = chunk_input_ids[:, 1:]

            # Use PyTorch's fused cross-entropy kernel.
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                teacher_targets.reshape(-1),
                reduction="none"
            ).view(chunk_input_ids.size(0), max_len - 1)

            # Keep only the action-token contribution.
            chunk_logprobs = -(loss * chunk_target_mask[:, :max_len-1]).sum(dim=1)
            action_logprobs.extend([s for s in chunk_logprobs])

        return action_logprobs

