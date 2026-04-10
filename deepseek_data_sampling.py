#!/usr/bin/env python3
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import random

from datasets import load_dataset
from betazero.policy.prompt import build_prompt
from betazero.policy.output_parser import get_lean_code
from betazero.utils.lean_cmd import DEFAULT_OPEN
from betazero.utils.lean_parse import parse_proof_state

# ============================================================================
# ⚙️ CẤU HÌNH (Sửa trực tiếp ở đây cho lẹ)
# ============================================================================
DATASET = "deepseek-ai/DeepSeek-Prover-V1"
SPLIT   = "train"
LIMIT   = 406           # Số lượng mẫu "HOÀN HẢO" mục tiêu
SHUFFLE = True          # Xáo trộn thứ tự câu hỏi?
SEED    = 1234

API_URL = "http://127.0.0.1:8787/chat"
ACTION  = "skeleton"  # "skeleton" hoặc "tactic"
MODE    = "Instant"   # DeepSeek mode

# API Timeouts (ms)
TIMEOUT_MS     = 240000
TURN_DELAY_MS  = 2000
PAUSE_LOGIN_MS = 1000

EXTRA_CRITICAL_RULES = (
    "\nCRITICAL RULES:\n"
    "Output in <think> </think> TAG:\n"
    "1. FOCUSED REASONING: Your <think> process MUST be step-by-step and deeply logical. Break the problem into clear milestones.\n"
    "2. SHOW YOUR WORK: Explicitly calculate algebraic steps, but DO NOT over-explain trivial arithmetic.\n"
    "3. DIRECT PATH: Stick strictly to the most promising mathematical path. DO NOT simulate endless alternative approaches or backtrack unnecessarily.\n"
)

@dataclass(frozen=True)
class ProblemResult:
    path: str
    header: str
    context: str
    goal: str
    prompt: str
    raw_output: str | None
    extracted_code: str

def get_deepseek_response(prompt: str) -> str | None:
    """Gọi API DeepSeek và bóc luôn cái response cuối cùng."""
    payload = {
        "mode": MODE, "pauseForLoginMs": PAUSE_LOGIN_MS,
        "turnDelayMs": TURN_DELAY_MS, "timeoutMs": TIMEOUT_MS,
        "prompts": [prompt],
    }
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=(TIMEOUT_MS // 1000) + 60) as resp:
            data = json.loads(resp.read().decode())
            return data.get("result", {}).get("turns", [{}])[-1].get("response")
    except Exception as e:
        print(f"❌ Lỗi gọi API: {e}")
        return None

def main():
    ds = load_dataset(DATASET)
    if SPLIT not in ds:
        return print(f"❌ Không tìm thấy split '{SPLIT}' trong dataset!")
    
    split_data = ds[SPLIT]
    indices = list(range(len(split_data)))
    if SHUFFLE: random.Random(SEED).shuffle(indices)

    idx_ptr = 0
    n_sent = 0
    while n_sent < LIMIT and idx_ptr < len(indices):
        idx = indices[idx_ptr]
        idx_ptr += 1
        
        row = split_data[idx]
        content = (row.get("formal_statement") or "").strip()
        if not content:
            continue
        
        # Lấy tên Theorem
        m_name = re.search(r"\btheorem\s+(\w+)\b", content)
        name = m_name.group(1) if m_name else (row.get("name") or f"row_{idx}")

        # Ép dùng `sorry` thay vì `by ...`
        content = re.sub(r":=\s*by\b.*", ":= by\n  sorry", content, flags=re.DOTALL)
        if "sorry" not in content:
            content += "\n  sorry\n"

        # Trộn header (import) sao cho chuẩn syntax
        prefix = (row.get("header") or "").strip() or DEFAULT_OPEN
        lines = content.splitlines()
        k = next((i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("import ")), len(lines))
        code_to_verify = "\n".join(lines[:k] + [prefix] + lines[k:]) if prefix else content

        # Build Prompt
        # Ở đây sẽ không verify Lean, chỉ lấy luôn goal đầu (lấy toàn bộ content làm context/goal nếu thiếu)
        try:
            goal_match = re.search(r"\{(.+)\}", code_to_verify, re.DOTALL)
            goal_str = goal_match.group(1) if goal_match else ""
        except Exception:
            goal_str = ""

        # In practice, parse_proof_state dựa vào goal string, header sẽ lấy từ prefix
        ps = parse_proof_state(goal_str, header=prefix)

        base_prompt = build_prompt(ps, action_type=ACTION)
        prompt = base_prompt.replace("<|im_start|>user\n", f"{EXTRA_CRITICAL_RULES}<|im_start|>user\n")

        print(f"[{n_sent+1}/{LIMIT}] {name} 🚀 Requesting...", end=" ", flush=True)
        print("✅ Done")
        n_sent += 1

    print(f"\n✅ Đã hoàn thành! Đã gửi {n_sent} request.")

if __name__ == "__main__":
    main()