import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from betazero.data.nodes import ProofState, Action
from betazero.logic.grpo_trainer import GRPOTrainer

# ==========================================
# 1. MODEL SIÊU NHỎ ĐỂ ÉP HỌC VẸT
# Dùng con gpt2 (124M params) - Vừa nhẹ vừa dễ train.
# ==========================================
class SmallPolicy(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        print(f"🚀 Đang load model bé xíu: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT2 không có pad_token_id, ta phải gán
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model có khả năng train
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def parameters(self):
        return self.model.parameters()

    def _build_prompt(self, state, action_type):
        # Thêm dấu cách sau \n cuối cùng để chống Tokenizer gộp chữ
        return f"State: {state.goal}\n\nInstruction: {action_type}\n\nCode:\n "

    def log_probs(self, states, actions, action_types):
        """Bê y nguyên logic dóng token và masking tao viết ban nãy vào đây."""
        full_texts = [
            self._build_prompt(s, atype) + act
            for s, act, atype in zip(states, actions, action_types)
        ]
        prompts = [self._build_prompt(s, atype) for s, atype in zip(states, action_types)]
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompt_lens = prompt_inputs.attention_mask.sum(dim=1)

        # Token hóa cả câu, pad bên phải cho train
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True)
        
        # Forward pass model đang học
        outputs = self.model(**inputs)
        logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

        # Dịch chéo (Shift)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()

        # Log softmax
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs_all, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Masking: Chỉ giữ lại phần Action
        batch_size, seq_len = shift_labels.shape
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        
        for i in range(batch_size):
            p_len = prompt_lens[i].item()
            start_idx = p_len - 1 
            end_idx = inputs.attention_mask[i].sum().item() - 1
            mask[i, start_idx:end_idx] = True

        # Trả về tổng LogProb của cả hành động
        return (token_log_probs * mask.to(token_log_probs.dtype)).sum(dim=1)

# ==========================================
# 2. MOCK ROLLOUT (Fix cứng 2 action tốt-xấu)
# ==========================================
class FixedRollout:
    def rollout(self, theorem: ProofState):
        s = ProofState("ctx_root", theorem.goal)

        a_good = Action("tactic", "rw [Nat.add_comm]", ())
        a_mid  = Action("tactic", "apply Nat.add_comm", ()) # Cùng ý nghĩa nhưng điểm thấp hơn 1 tí

        return [
            (s, a_good, 1.0, 0.0), 
            (s, a_mid,  0.5, 0.0), 
        ]

# ==========================================
# 3. KỊCH BẢN CHẠY THỬ
# ==========================================
def run_overfit_test():
    policy = SmallPolicy()
    ref_policy = SmallPolicy() # Giống hệt policy, nhưng sẽ bị trói tay
    
    # Copy trọng số sang ref_policy
    ref_policy.load_state_dict(policy.state_dict())
    
    # Lấy state và action ra để soi độ tự tin qua từng epoch
    s_test = ProofState("ctx_root", "a + b = b + a")
    a_good_text = "rw [Nat.add_comm]"
    a_mid_text = "apply Nat.add_comm"

    rollout = FixedRollout()

    # Tạo Trainer
    trainer = GRPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        rollout=rollout,
        lr=5e-6, # LR to để model học lẹ hơn
        eps_clip=0.2,
        beta_kl=0.0,
        epochs_per_batch=1, # Mỗi batch chỉ train 1 lần, ta sẽ tự lặp batch bên ngoài
        mini_batch_size=2 # Train cả mẻ
    )

    print("\n" + "="*40)
    print("🏃 BẮT ĐẦU ÉP MODEL HỌC VẸT (1000 EPOCHS)")
    print("="*40)
    
    print(f"{'Epoch':<8} | {'Loss':<10} | {'LogProb(Good)':<15} | {'LogProb(Mid)':<15}")
    print("-" * 60)

    for epoch in range(1, 1001):
        # 1. Đo độ tự tin của model TRƯỚC khi update
        with torch.no_grad():
            lp_good = policy.log_probs([s_test], [a_good_text], ["tactic"])[0].item()
            lp_mid = policy.log_probs([s_test], [a_mid_text], ["tactic"])[0].item()

        # 2. Chạy hàm train! 
        dummy_theorem = ProofState("ctx_root", "a + b = b + a")
        metrics = trainer.train([dummy_theorem])

        # ========================================================
        # TRÒ ẢO THUẬT: Cập nhật Reference Model liên tục!
        # Reset ratio về 1.0 để qua epoch sau model không bị chạm trần
        # ========================================================
        ref_policy.load_state_dict(policy.state_dict())

        # 3. In log
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} | {metrics['loss']:<10.4f} | {lp_good:<15.4f} | {lp_mid:<15.4f}")

    print("-" * 60)
    print("\n✅ THÀNH CÔNG: Model đã overfit!")
    print("   -> Bạn phải thấy LogProb(Good) TĂNG lên (về gần 0) và LogProb(Mid) TỤT xuống.")
    print("   -> Nếu 2 con số này đứng im, thì gradient bị đứt gãy đâu đó!")

if __name__ == "__main__":
    run_overfit_test()