import os
import gc
import torch
from collections import defaultdict
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

        # Load reference adapters
        self.model = PeftModel.from_pretrained(base, base_tactic, adapter_name="ref_tactic")
        if os.path.exists(base_skeleton):
            self.model.load_adapter(base_skeleton, adapter_name="ref_skeleton")

        # Load or initialize Active RL adapters
        if adapter_path and os.path.exists(adapter_path):
            tactic_path = os.path.join(adapter_path, "tactic")
            skeleton_path = os.path.join(adapter_path, "skeleton")
            if os.path.exists(tactic_path):
                self.model.load_adapter(tactic_path, adapter_name="active_tactic", is_trainable=True)
            if os.path.exists(skeleton_path):
                self.model.load_adapter(skeleton_path, adapter_name="active_skeleton", is_trainable=True)
        else:
            # At iteration 1, duplicate the SFT base but make them trainable for RL
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
        """
        Tính log_probs bằng cách nối thẳng Prompt + Action.
        Bỏ Shared Prefill, tận dụng FlashAttention-2 để PyTorch tự cấp phát và backprop chuẩn xác.
        """
        if not prompts:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        # Gom nhóm index theo prompt và action_type để switch adapter
        group_to_idxs = defaultdict(list)
        for i, (p, t) in enumerate(zip(prompts, action_types)):
            group_to_idxs[(p, t)].append(i)

        scores = torch.zeros(len(prompts), dtype=torch.float32, device=self.device)
        
        # Bắt buộc lưu lại padding_side cũ và set right padding để tính toán index chính xác
        orig_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        try:
            for (p_text, a_type), idxs in group_to_idxs.items():
                group_actions = [actions[i] for i in idxs]
                
                # Chuyển đổi Multi-Adapter
                target_adapter = f"ref_{a_type}" if disable_adapter else f"active_{a_type}"
                self.model.set_adapter(target_adapter)
                
                # Gọi hàm tính Logprob đơn giản
                group_scores = self._score_direct(p_text, group_actions)
                
                for i, score in zip(idxs, group_scores):
                    scores[i] = score
        finally:
            self.tokenizer.padding_side = orig_padding

        return scores

    def _score_direct(self, prompt: str, actions: list[str]) -> list[torch.Tensor]:
        """Tính Log Prob của các đoạn Action bằng cách tự nối ID (An toàn 100% không bị BPE merge)."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        # Tokenize các râu (actions) riêng biệt
        encoded_actions = self.tokenizer(actions, add_special_tokens=False)

        # Nối ID thủ công và đệm (Right Padding)
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

        # Forward một cách an toàn
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # (B, T, V)

        target_mask = torch.zeros_like(attention_mask)
        
        # Mask chỉ trùm lên đúng token đầu tiên của Action trở thành Target
        start_idx = prompt_len - 1  # Logit t dự đoán t+1
        for i in range(B):
            end_idx = attention_mask[i].sum() - 1
            if end_idx > start_idx:
                target_mask[i, start_idx:end_idx] = 1

        teacher_targets = input_ids[:, 1:]
        logits = logits[:, :-1, :]

        # Dùng Fused C++ Kernel siêu tốc của Pytorch
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), 
            teacher_targets.reshape(-1), 
            reduction="none"
        ).view(B, max_len - 1)
        
        # Chỉ giữ lại phần log_prob của action
        action_logprobs = -(loss * target_mask[:, :max_len-1]).sum(dim=1)
        
        return [s for s in action_logprobs]
