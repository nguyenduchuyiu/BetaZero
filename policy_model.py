import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nodes import ProofState

class PolicyModel:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        # Load tokenizer và cấu hình padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Generate thì pad bên trái

        # Load model với khả năng tính gradient để train
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          dtype=torch.bfloat16,
                                                          device_map=device)

    def parameters(self):
        """Dễ nhất quả đất, ném cho optimizer là xong."""
        return self.model.parameters()

    def _build_prompt(self, state: ProofState, action_type: str) -> str:
        """Hàm này cực kỳ quan trọng: Ép LLM đẻ ra đúng loại code"""
        instruction = "Write a Lean 4 tactic." if action_type == "tactic" \
                 else "Write a Lean 4 skeleton proof using 'sorry'."
        
        # Mày có thể đổi template này tùy theo cách mày pre-train model
        prompt = f"Context:\n{state.context}\n\nGoal:\n{state.goal}\n\nInstruction: {instruction}\n\nCode:\n"
        return prompt
    
    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]:
        prompts = [self._build_prompt(s, action_type) for s in states]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        # [SỬA LỖI LEFT-PADDING]: Lấy chiều dài trục 1 của ma trận input
        prompt_max_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id
            )

        batch_size = len(states)
        results = []
        for i in range(batch_size):
            state_actions = []
            for j in range(n):
                idx = i * n + j
                # Cắt một nhát gọn gàng từ điểm kết thúc của phần Input Batch
                gen_tokens = outputs[idx][prompt_max_len:]
                
                action_str = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                state_actions.append(action_str.strip())
            results.append(state_actions)

        return results
    
    def log_probs(self, states: list[ProofState], actions: list[str],
                  action_types: list[str]) -> torch.Tensor:
        """Tính log pi(a|s) để update gradient."""
        
        # 1. Ghép prompt và action lại thành 1 câu hoàn chỉnh
        full_texts = [
            self._build_prompt(s, atype) + act
            for s, act, atype in zip(states, actions, action_types)
        ]
        
        # Để dễ bóc tách, token hóa riêng phần prompt để lấy độ dài
        prompts = [self._build_prompt(s, atype) for s, atype in zip(states, action_types)]
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompt_lens = prompt_inputs.attention_mask.sum(dim=1)

        # Token hóa cả câu
        self.tokenizer.padding_side = "right" # Lúc train thì pad bên phải cho chuẩn
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
        
        # 2. Forward pass lấy Logits (Không có no_grad ở đây vì cần backward)
        outputs = self.model(**inputs)
        logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

        # 3. Tính toán Log Probs theo chuẩn Causal LM (next token prediction)
        # Bỏ token cuối của logits, bỏ token đầu của labels để khớp nhau
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        # Tính log softmax cho toàn bộ từ điển
        log_probs_all = F.log_softmax(shift_logits, dim=-1)

        # Bốc đúng xác suất của cái token thực tế đã sinh ra
        # Shape: (batch, seq_len-1)
        token_log_probs = torch.gather(log_probs_all, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # 4. Masking: Che đi phần prompt và phần padding, chỉ giữ lại phần Action
        batch_size, seq_len = shift_labels.shape
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        
        for i in range(batch_size):
            p_len = prompt_lens[i].item()
            # Bắt đầu tính từ token ngay sau prompt
            start_idx = p_len - 1 
            # Kết thúc tại token thực tế cuối cùng (bỏ qua padding)
            end_idx = inputs.attention_mask[i].sum().item() - 1
            mask[i, start_idx:end_idx] = True

        # 5. Tổng hợp (Sum) log probs của từng token thành log prob của cả hành động
        action_log_probs = (token_log_probs * mask.to(token_log_probs.dtype)).sum(dim=1)

        return action_log_probs # Shape: (batch,)