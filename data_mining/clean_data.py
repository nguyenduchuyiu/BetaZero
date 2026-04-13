#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Chuỗi cần xóa (Copy y hệt từ deepseek_data_sampling.py)
EXTRA_CRITICAL_RULES = (
    "\nCRITICAL RULES:\n"
    "1. FOCUSED REASONING: Your <think> process MUST be step-by-step and deeply logical. Break the problem into clear milestones.\n"
    "2. SHOW YOUR WORK: Explicitly calculate algebraic steps, but DO NOT over-explain trivial arithmetic.\n"
    "3. DIRECT PATH: Stick strictly to the most promising mathematical path. DO NOT simulate endless alternative approaches or backtrack unnecessarily.\n"
)
EXTRA_EXAMPLE = (
    "---EXAMPLE ---\n"
    "[INPUT]\n"
    "import Mathlib\n\n"
    "open Function\n\n"
    "theorem challenge_01 (f : ℤ → ℤ) (a b c : ℤ)\n"
    "  (h_inj : Injective f)\n"
    "  (h1 : f (2 * a + 3) = f (b - 1))\n"
    "  (h2 : b < 10)\n"
    "  (h3 : c > a ^ 2 + 5)\n"
    "  (h4 : f 0 = 0) : 2 * a < 8 := by sorry\n\n"
    "[EXPECTED OUTPUT]\n"
    "<think>\n"
    "(Example of filtering the noise)\n"
    "Goal: `2 * a < 8`.\n"
    "Relevant hypotheses: `h_inj` (`Injective f`), `h1` (`f (2 * a + 3) = f (b - 1)`), and `h2` (`b < 10`).\n"
    "Irrelevant noise: `c`, `h3`, and `h4` (ignore these distractors).\n"
    "Strategy: Use the injectivity of `f` (`h_inj`) on `h1` to extract the equality `2 * a + 3 = b - 1`. This simplifies to `2 * a = b - 4`. Since `b < 10` (`h2`), we can deduce `2 * a < 10 - 4 = 6`, which strictly implies `2 * a < 8`. We can introduce the equality via `have` and then use `omega` to handle the linear arithmetic automatically.\n"
    "</think>\n"
    "```lean4\n"
    "import Mathlib\n\n"
    "open Function\n\n"
    "theorem challenge_01 (f : ℤ → ℤ) (a b c : ℤ)\n"
    "  (h_inj : Injective f)\n"
    "  (h1 : f (2 * a + 3) = f (b - 1))\n"
    "  (h2 : b < 10)\n"
    "  (h3 : c > a ^ 2 + 5)\n"
    "  (h4 : f 0 = 0) : 2 * a < 8 := by\n"
    "  have h_eq := h_inj h1\n"
    "  omega\n"
    "```"
)

SKELETON_EXAMPLE = (
    "### EXAMPLE\n\n"
    "[INPUT]\n"
    "```lean4\n"
    "import Mathlib\n"
    "open Nat\n"
    "theorem my_theorem (n : ℕ) : 6 ∣ n^3 - n := by sorry\n"
    "````\n\n"
    "[EXPECTED OUTPUT]\n"
    "<think>\n"
    "(Your thinking process here)\n"
    "</think>\n"
    "```lean4\n"
    "theorem my_theorem (n : ℕ) : 6 ∣ n^3 - n := by\n"
    "  have h2 : 2 ∣ n^3 - n := sorry\n"
    "  have h3 : 3 ∣ n^3 - n := sorry\n"
    "  have h_coprime : Nat.Coprime 2 3 := sorry\n"
    "  have h_combine : 2 * 3 ∣ n^3 - n := sorry\n"
    "  have h_final : 6 ∣ n^3 - n := sorry\n"
    "  exact h_final\n"
    "```"
)

def clean_content(content):
    if isinstance(content, str):
        # Ưu tiên xóa các đoạn ví dụ dài trước
        cleaned = content.replace(EXTRA_EXAMPLE.strip(), "")
        cleaned = cleaned.replace(SKELETON_EXAMPLE.strip(), "")
        cleaned = cleaned.replace(EXTRA_CRITICAL_RULES.strip(), "")
        return cleaned.strip()
    elif isinstance(content, list):
        return [clean_content(i) for i in content]
    elif isinstance(content, dict):
        return {k: clean_content(v) for k, v in content.items()}
    return content

def main():
    if len(sys.argv) < 2:
        print("💡 Cách dùng: python clean_data.py <path_to_json_file>")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print(f"🧹 Đang làm sạch file: {file_path}...")
    
    cleaned_items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cleaned_items.append(clean_content(data))

    # Lưu đè lên file cũ dưới dạng JSONL
    with open(file_path, "w", encoding="utf-8") as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã dọn dẹp xong! File {file_path} hiện đã sạch bóng CRITICAL RULES (JSONL format).")

if __name__ == "__main__":
    main()
