#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để import được betazero
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from betazero.env.lean_verifier import Lean4ServerScheduler


ROOT = Path(__file__).resolve().parent.parent
SFT_FILE = ROOT / "data" / "tactic_samples.jsonl"
OUT_PASS = ROOT / "data" / "passed_tactics.jsonl"
OUT_FAIL = ROOT / "data" / "failed_tactics.jsonl"
from betazero.utils.lean_cmd import build_theorem
from betazero.utils.lean_parse import parse_proof_state


def main():
    if not SFT_FILE.exists():
        print(f"Khong tim thay {SFT_FILE}")
        return

    samples = []
    with SFT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    passed, failed = [], []
    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=120, name="verify_sft")

    print(f"Verifying {len(samples)} samples with strict reconstruction...")
    try:
        for i, sample in enumerate(samples, start=1):
            theorem = sample.get("theorem", "")
            header = sample.get("header", "")
            raw_output = sample.get("raw_output", "")
            extracted_code = sample.get("extracted_code", "")
            thm_name = sample.get("id", f"sample_{i}")

            # 1. Bóc ProofState gốc (Rào lỗi signature ngay từ đây)
            vr_init = sch.verify(theorem)
            if not vr_init.get("pass") or not vr_init.get("sorries"):
                err = vr_init.get("errors", [{}])[0].get("data", "Init Failed")
                sample["error_reason"] = f"Original Theorem Error: {err[:100]}"
                failed.append(sample)
                print(f"[{i}/{len(samples)}] {thm_name} FAIL init")
                continue
            
            ps_init = parse_proof_state(vr_init["sorries"][0]["goal"], header=header)

            # 2. Xây dựng lại toàn bộ theorem bằng build_theorem (Rào lỗi shadowing/xung đột)
            full_code = build_theorem(ps_init, extracted_code, name=thm_name)


            # 4. Verify chứng minh cuối cùng
            vr = sch.verify(full_code)
            if vr.get("complete"):
                sample["error_reason"] = ""
                passed.append(sample)
                print(f"[{i}/{len(samples)}] {thm_name} COMPLETE")
            else:
                errors = vr.get("errors", [])
                sorries = vr.get("sorries", [])
                
                if errors:
                    err = errors[0].get("data", "Unknown Lean Error")
                    sample["error_reason"] = f"Lean Error: {err}"
                    print(f"[{i}/{len(samples)}] {thm_name} FAIL lean")
                elif sorries:
                    sample["error_reason"] = "Unsolved goals (sorry)"
                    print(f"[{i}/{len(samples)}] {thm_name} FAIL incomplete")
                else:
                    err = "Unknown error or sorry warning"
                    sample["error_reason"] = f"Lean Error: {err}"
                    print(f"[{i}/{len(samples)}] {thm_name} FAIL other")
                
                failed.append(sample)

    finally:
        sch.close()

    OUT_PASS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PASS.open("w", encoding="utf-8") as f:
        for s in passed:
            s.pop("error_reason", None)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with OUT_FAIL.open("w", encoding="utf-8") as f:
        for s in failed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Done. pass={len(passed)} fail={len(failed)}")
    print(f"Saved: {OUT_PASS}, {OUT_FAIL}")


if __name__ == "__main__":
    main()
