import os
import gc
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from betazero.core import ProofState
from betazero.policy.prompt import build_prompt
from betazero.utils import Config


class TrainablePolicy:
    """PEFT model for log_probs + gradient update. Explicitly load/unload to free VRAM."""

    def __init__(self, cfg: Config, adapter_path: str | None = None):
        self.device = cfg.device
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

    def log_probs(self, states: list[ProofState], actions: list[str],
                  action_types: list[str],
                  disable_adapter: bool = False) -> torch.Tensor:
        """Sum log pi(a|s) over action tokens. disable_adapter=True → base model (ref policy)."""
        prompts    = [build_prompt(s, at) for s, at in zip(states, action_types)]
        full_texts = [p + a for p, a in zip(prompts, actions)]

        prompt_lens = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).attention_mask.sum(dim=1)

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
        self.tokenizer.padding_side = "left"

        ctx = self.model.disable_adapter() if disable_adapter else nullcontext()
        with ctx:
            logits = self.model(**inputs).logits

        mask = torch.zeros_like(inputs.input_ids[:, 1:], dtype=torch.bool)
        for i in range(len(states)):
            mask[i, prompt_lens[i].item() - 1: inputs.attention_mask[i].sum().item() - 1] = True

        min_prompt_len = prompt_lens.min().item()
        start_idx = min_prompt_len - 1
        
        rel_logits = logits[:, start_idx:-1, :]  # [B, T_action, V]
        rel_targets = inputs.input_ids[:, start_idx+1:].unsqueeze(-1) # [B, T_action, 1]
        rel_mask = mask[:, start_idx:] # [B, T_action]

        tok_logits = torch.gather(rel_logits, dim=2, index=rel_targets).squeeze(-1)
        lse = torch.logsumexp(rel_logits, dim=-1) 
        rel_token_lp = tok_logits.float() - lse

        return (rel_token_lp * rel_mask).sum(dim=1)
