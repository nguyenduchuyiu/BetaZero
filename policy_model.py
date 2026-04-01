import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from nodes import ProofState
from config import Config


class PolicyModel:
    def __init__(self, cfg: Config):
        self.device = cfg.device
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature    = cfg.temperature

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map=cfg.device
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()

    def parameters(self):
        return self.model.parameters()

    def _build_prompt(self, state: ProofState, action_type: str) -> str:
        instruction = "Write a Lean 4 tactic." if action_type == "tactic" \
                 else "Write a Lean 4 skeleton proof using 'sorry'."
        return f"Context:\n{state.context}\n\nGoal:\n{state.goal}\n\nInstruction: {instruction}\n\nCode:\n"

    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]:
        prompts = [self._build_prompt(s, action_type) for s in states]
        inputs  = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i in range(len(states)):
            results.append([
                self.tokenizer.decode(outputs[i * n + j][prompt_len:], skip_special_tokens=True).strip()
                for j in range(n)
            ])
        return results

    def log_probs(self, states: list[ProofState], actions: list[str],
                  action_types: list[str]) -> torch.Tensor:
        """Compute sum log pi(a|s) over action tokens only."""
        prompts    = [self._build_prompt(s, at) for s, at in zip(states, action_types)]
        full_texts = [p + a for p, a in zip(prompts, actions)]

        prompt_lens = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).attention_mask.sum(dim=1)

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
        self.tokenizer.padding_side = "left"

        logits = self.model(**inputs).logits
        log_probs_all  = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_lp = torch.gather(
            log_probs_all, dim=2,
            index=inputs.input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Mask: keep only action tokens (after prompt, before padding)
        mask = torch.zeros_like(inputs.input_ids[:, 1:], dtype=torch.bool)
        for i in range(len(states)):
            start = prompt_lens[i].item() - 1
            end   = inputs.attention_mask[i].sum().item() - 1
            mask[i, start:end] = True

        return (token_lp * mask).sum(dim=1)
