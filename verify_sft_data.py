#!/usr/bin/env python3
import json
import re
from pathlib import Path

from betazero.env.lean_verifier import Lean4ServerScheduler

SFT_FILE = Path("automation-output/SFT_data.json")
OUT_PASS = Path("automation-output/verified_passed.json")
OUT_FAIL = Path("automation-output/verified_failed.json")

FORBIDDEN_STARTS = ["cases ", "induction ", "match ", "by_cases ", "obtain "]
ALLOWED_HAVE = re.compile(
    r":=\s*("
    # --- GROUP 1: TACTIC MODE ---
    # Đã bổ sung rfl, sorry, trivial, ring_nf. Vẫn tuyệt đối cấm dấu chấm phẩy ;
    r"(by\s+(ring_nf|ring|linarith|norm_num|exact|rw|erw|field_simp|apply|positivity|qarith|refine|simp_all|simp|dsimp|aesop|omega|nlinarith|rfl|sorry|trivial)(?!.*;).*)"
    r"|"
    # --- GROUP 2: TERM MODE ---
    # Vẫn cấm dấu ;
    # Cho phép chữ 'by' NẾU NÓ NẰM SAU DẤU NGOẶC (Ví dụ: (by norm_num)). Cấm 'by' tự do.
    r"(sorry|rfl|trivial|⟨.*?⟩|fun\s+.*?=>.*?|(?!.*(?<!\()\bby\b)[^;]+)"
    r")\s*$"
)


def build_full_code(theorem: str, extracted_code: str) -> str:
    m_by = re.search(r":=\s*by\b", theorem)
    if m_by:
        indented = "\n".join("  " + line for line in extracted_code.splitlines())
        return theorem[: m_by.end()] + "\n" + indented
    return theorem.replace("sorry", extracted_code)


def is_perfect_skeleton(raw: str, code: str) -> tuple[bool, str]:
    if not code.strip():
        return False, "Empty code block"

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match and len(think_match.group(1)) // 4 > 4000:
        return False, "Reasoning too long (>4000 tokens)"

    for line in code.splitlines():
        clean_line = line.split("--")[0].strip()
        if not clean_line:
            continue
        if any(clean_line.startswith(fs) for fs in FORBIDDEN_STARTS) or "= calc" in clean_line or clean_line.startswith("calc "):
            return False, f"Forbidden structure/start in line: {clean_line}"
        if clean_line.startswith("have "):
            if ":" not in clean_line:
                return False, f"Missing type annotation in have: {clean_line}"
            if not ALLOWED_HAVE.search(clean_line):
                return False, f"Forbidden have syntax: {clean_line}"
    return True, ""


def main():
    if not SFT_FILE.exists():
        print(f"Khong tim thay {SFT_FILE}")
        return

    with SFT_FILE.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    passed, failed = [], []
    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=60, name="verify_sft")

    print(f"Verifying {len(samples)} samples...")
    try:
        for i, sample in enumerate(samples, start=1):
            theorem = sample.get("theorem", "")
            raw_output = sample.get("raw_output", "")
            extracted_code = sample.get("extracted_code", "")
            full_code = build_full_code(theorem, extracted_code)

            ok, reason = is_perfect_skeleton(raw_output, full_code)
            if not ok:
                sample["error_reason"] = f"Gate Failed: {reason}"
                failed.append(sample)
                print(f"[{i}/{len(samples)}] {sample.get('id', f'sample_{i}')} FAIL gate")
                continue

            vr = sch.verify(full_code)
            if vr.get("pass"):
                sample["error_reason"] = ""
                passed.append(sample)
                print(f"[{i}/{len(samples)}] {sample.get('id', f'sample_{i}')} PASS")
            else:
                errors = vr.get("errors", [])
                err = errors[0].get("data", "Unknown Lean Error") if errors else "REPL Panic"
                sample["error_reason"] = f"Lean Error: {err}"
                failed.append(sample)
                print(f"[{i}/{len(samples)}] {sample.get('id', f'sample_{i}')} FAIL lean")
    finally:
        sch.close()

    OUT_PASS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PASS.open("w", encoding="utf-8") as f:
        json.dump(passed, f, indent=2, ensure_ascii=False)
    with OUT_FAIL.open("w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    print(f"Done. pass={len(passed)} fail={len(failed)}")
    print(f"Saved: {OUT_PASS}, {OUT_FAIL}")


if __name__ == "__main__":
    main()
