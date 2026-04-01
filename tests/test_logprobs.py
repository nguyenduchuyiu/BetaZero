import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mock cái ProofState cho code chạy được
class ProofState:
    def __init__(self, context, goal):
        self.context = context
        self.goal = goal

def test_deepseek_log_probs():
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_name = "gpt2"
    print(f"🚀 Đang load Tokenizer và Model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load model với bfloat16 và auto device map để không chết RAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda"
    )

    # 1. Khởi tạo data giả lập
    state = ProofState(context="n : Nat", goal="n + 0 = n")
    action_type = "tactic"
    action_code = "rw [Nat.add_zero]"

    # Build prompt y như trong HuggingFacePolicy
    prompt = f"Context:\n{state.context}\n\nGoal:\n{state.goal}\n\nInstruction: Write a Lean 4 tactic.\n\nCode:\n"
    full_text = prompt + action_code

    print("\n" + "="*40)
    print("🔍 KIỂM TRA ĐỘ DÀI TOKENIZER")
    print("="*40)
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
    
    prompt_len = len(prompt_tokens)
    print(f"Độ dài Prompt: {prompt_len} tokens")
    print(f"Độ dài Full Text: {len(full_tokens)} tokens")
    print(f"-> Action Code chiếm đúng: {len(full_tokens) - prompt_len} tokens")

    # 2. Đưa vào Model tính toán (Mock hàm log_probs)
    inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
    
    with torch.no_grad(): # Test thì no_grad cho lẹ
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. Dịch chéo (Shift) để tính next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()

    # Tính xác suất
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs_all, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # 4. Masking: Cắt đúng phần Action
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    
    # Do đã shift 1 token, index bắt đầu của Action trên shift_labels sẽ lùi lại 1
    start_idx = prompt_len - 1 
    end_idx = inputs.attention_mask.sum().item() - 1
    mask[0, start_idx:end_idx] = True

    print("\n" + "="*40)
    print("✂️ BÓC TÁCH LOG PROB CỦA TỪNG TOKEN TRONG ACTION")
    print("="*40)
    
    total_log_prob = 0.0
    action_token_count = 0

    for i in range(len(shift_labels[0])):
        if mask[0, i]:
            token_id = shift_labels[0, i].item()
            token_str = tokenizer.decode([token_id])
            prob = token_log_probs[0, i].item()
            
            total_log_prob += prob
            action_token_count += 1
            print(f"Token [{action_token_count}]: {token_str!r:<15} | LogProb: {prob:.4f}")

    print("-" * 40)
    print(f"🎯 TỔNG LOG PROB (Tensor trả về cho GRPO): {total_log_prob:.4f}")
    
    # Verify lại số lượng token cắn đúng không
    if action_token_count == (len(full_tokens) - prompt_len):
        print("✅ PASSED: Mask bắt chuẩn xác 100% phần Action, không cắn nhầm vào Prompt!")
    else:
        print("❌ FAILED: Lệch index do Tokenizer nối chữ (Merge space)!")

if __name__ == '__main__':
    test_deepseek_log_probs()