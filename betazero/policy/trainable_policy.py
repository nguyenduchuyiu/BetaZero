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