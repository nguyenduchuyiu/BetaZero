#!/usr/bin/env python3
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import random

from datasets import load_dataset
from betazero.policy.prompt import build_prompt
from betazero.policy.output_parser import get_lean_code
from betazero.utils.lean_cmd import DEFAULT_OPEN
from betazero.utils.lean_parse import parse_proof_state
from betazero.env.lean_verifier import Lean4ServerScheduler

# ============================================================================
# ⚙️ CẤU HÌNH (Sửa trực tiếp ở đây cho lẹ)
# ============================================================================
DATASET = "deepseek-ai/DeepSeek-Prover-V1"
SPLIT   = "train"
LIMIT   = 1           # Giới hạn số câu chạy (0 = chạy hết)
SHUFFLE = True       # Xáo trộn thứ tự câu hỏi?
SEED    = 42

API_URL = "http://127.0.0.1:18787/chat"
ACTION  = "skeleton"  # "skeleton" hoặc "tactic"
MODE    = "Instant"   # DeepSeek mode
OUT_DIR = Path("automation-output/deepseek_results.json")

# API Timeouts (ms)
TIMEOUT_MS     = 240000
TURN_DELAY_MS  = 2000
PAUSE_LOGIN_MS = 1000

# Lean Verifier Config
LEAN_TIMEOUT_S = 60
PRINT_ON_ERROR = True  # Có in code lỗi ra màn hình không?

EXTRA_CRITICAL_RULES = (
    "\nCRITICAL RULES:\n"
    "Output in <think> </think> TAG:\n"
    "1. FOCUSED REASONING: Your <think> process MUST be step-by-step and deeply logical. Break the problem into clear milestones.\n"
    "2. SHOW YOUR WORK: Explicitly calculate algebraic steps, but DO NOT over-explain trivial arithmetic.\n"
    "3. DIRECT PATH: Stick strictly to the most promising mathematical path. DO NOT simulate endless alternative approaches or backtrack unnecessarily.\n"
)
# ============================================================================

@dataclass(frozen=True)
class ProblemResult:
    path: str
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
    if LIMIT > 0: indices = indices[:LIMIT]

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    results = []
    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=LEAN_TIMEOUT_S, name="ds_sampling")

    try:
        for j, idx in enumerate(indices, 1):
            row = split_data[idx]
            content = (row.get("formal_statement") or "").strip()
            if not content: continue
            
            # Lấy tên Theorem
            m_name = re.search(r"\btheorem\s+(\w+)\b", content)
            name = m_name.group(1) if m_name else (row.get("name") or f"row_{idx}")

            # Ép dùng `sorry` thay vì `by ...`
            content = re.sub(r":=\s*by\b.*", ":= by\n  sorry", content, flags=re.DOTALL)
            if "sorry" not in content:
                content += "\n  sorry\n"

            # Trộn header (import) sao cho chuẩn syntax (Nhét Prefix vào sau import, trước theorem)
            prefix = (row.get("header") or "").strip() or DEFAULT_OPEN
            lines = content.splitlines()
            k = next((i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("import ")), len(lines))
            code_to_verify = "\n".join(lines[:k] + [prefix] + lines[k:]) if prefix else content

            # Kiểm duyệt Lean
            vr = sch.verify(code_to_verify)
            sorries = vr.get("sorries", [])
            
            if not sorries:
                if vr.get("errors"):
                    err_msg = str(vr["errors"][0].get("data", "")).replace("\n", " ")[:120]
                    print(f"[{j}/{len(indices)}] {name} ⏭️ SKIP (Lean Error: {err_msg})")
                    if PRINT_ON_ERROR: print(f"--- CODE ---\n{code_to_verify}\n------------")
                else:
                    print(f"[{j}/{len(indices)}] {name} ⏭️ SKIP (Không thấy lỗ sorry nào)")
                continue

            # Build Prompt & Tiêm Rule bằng lệnh string replace đơn giản
            goal_str = sorries[0].get("goal", "")
            ps = parse_proof_state(goal_str, header=prefix)
            
            base_prompt = build_prompt(ps, action_type=ACTION)
            prompt = base_prompt.replace("<|im_start|>user\n", f"{EXTRA_CRITICAL_RULES}<|im_start|>user\n")

            print(f"[{j}/{len(indices)}] {name} 🚀 Requesting...", flush=True)
            raw_output = get_deepseek_response(prompt)
            
            results.append(ProblemResult(
                path=f"{DATASET}:{SPLIT}:{name}",
                context=ps.context or "",
                goal=ps.goal or "",
                prompt=prompt,
                raw_output=raw_output,
                extracted_code=get_lean_code(raw_output or "")
            ))
            
    finally:
        sch.close()

    # Ghi File JSON
    with open(OUT_DIR, "w", encoding="utf-8") as f:
        json.dump({
            "api_url": API_URL, "action": ACTION, "dataset": DATASET, "split": SPLIT,
            "n_total": len(split_data), "n_run": len(indices), "n_parsed": len(results),
            "results": [asdict(r) for r in results]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Đã lưu {len(results)} kết quả vào {OUT_DIR}")

if __name__ == "__main__":
    main()