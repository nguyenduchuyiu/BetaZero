from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ==========================================
# ⚙️ 1. CẤU HÌNH SKELETON
# ==========================================
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "microsoft/Phi-4-mini-reasoning"
DATA_PATH = "data/passed_skeletons.jsonl"
OUTPUT_DIR = "lora_skeleton" # Thư mục lưu riêng cho Skeleton
MAX_SEQ_LENGTH = 4096

# ==========================================
# 🧠 2. LOAD MODEL & ÁP LORA
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ==========================================
# 📦 3. CHUẨN BỊ DATA SKELETON
# ==========================================
def format_data(examples):
    texts = []
    for p, r in zip(examples["prompt"], examples["raw_output"]):
        text = f"{p.rstrip()}{r.lstrip()}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=DATA_PATH, split="train").shuffle(seed=42)
dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

# ==========================================
# 🚀 4. HUẤN LUYỆN
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
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_torch",
        output_dir=OUTPUT_DIR,
    ),
)

print("🔥 BẮT ĐẦU HUẤN LUYỆN SKELETON ADAPTER...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Đã lưu Skeleton LoRA tại {OUTPUT_DIR}")