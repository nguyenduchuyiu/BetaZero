from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ==========================================
# ⚙️ 1. CẤU HÌNH
# ==========================================
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_PATH = "automation-output/verified_passed.json"
OUTPUT_DIR = "lora_skeleton_model"

MAX_SEQ_LENGTH = 4096 # Độ dài tối đa (Token)

# ==========================================
# 🧠 2. LOAD MODEL & ÁP LORA
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
)

# Gắn "vỏ" LoRA vào model gốc
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Rank 16 là điểm ngọt cho format
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ==========================================
# 📦 3. CHUẨN BỊ DATA
# ==========================================
# prompt đã là full chat prompt, nối raw_output thành 1 sample hoàn chỉnh
def format_data(examples):
    texts = []
    for p, r in zip(examples["prompt"], examples["raw_output"]):
        p = p.rstrip()
        r = r.lstrip()
        text = f"{p}\n{r}"
        if not text.endswith("<|im_end|>"):
            text = f"{text}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# ==========================================
# 🚀 4. KHỞI ĐỘNG TRAINER
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2, # Ép format thì chạy 1-2 Epoch là dư sức
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit", # Optimizer xịn, ăn ít RAM
        output_dir=OUTPUT_DIR,
    ),
)

print("🔥 BẮT ĐẦU TRAIN SFT LORA...")
trainer.train()

# Lưu trọng số LoRA (chỉ vài chục MB)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Đã lưu LoRA tại {OUTPUT_DIR}")