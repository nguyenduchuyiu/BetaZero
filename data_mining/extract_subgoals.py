#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để import được betazero
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT_DIR = Path(__file__).parent
os.chdir(ROOT_DIR)

import json
import re
from dataclasses import asdict
from betazero.utils.lean_cmd import build_theorem
from betazero.utils.lean_parse import parse_proof_state
from betazero.env.lean_verifier import Lean4ServerScheduler
from betazero.core.nodes import ProofState

# ============================================================================
# ⚙️ CẤU HÌNH
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "passed_skeletons.jsonl"
OUTPUT_FILE = ROOT / "data" / "unsolved_subgoals.jsonl"



def main():
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file {INPUT_FILE}")
        return

    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Khởi tạo Lean Server
    sch = Lean4ServerScheduler(max_concurrent_requests=1, timeout=120, name="subgoal_extractor")
    
    try:
        enriched_results = []
        total = len(data)

        print(f"🔄 Bắt đầu bóc tách subgoal cho {total} mẫu (StrictMode)...")

        for i, item in enumerate(data):
            thm_name = item.get("id", f"thm_{i}")
            header = item.get("header", "")
            theorem_decl = item.get("theorem", "")
            skeleton_code = item.get("extracted_code", "")

            # 1. Bóc ProofState gốc (Rào lỗi signature)
            vr_init = sch.verify(theorem_decl)
            if not vr_init.get("pass") or not vr_init.get("sorries"):
                print(f"[{i+1}/{total}] {thm_name} ⚠️ Lỗi signature gốc. Bỏ qua.")
                continue
                
            ps_init = parse_proof_state(vr_init["sorries"][0]["goal"], header=header)

            # 2. Xây dựng lại code bằng build_theorem (Rào lỗi shadowing)
            full_code = build_theorem(ps_init, skeleton_code, name=thm_name)

            # 3. Verify để lấy các subgoal tại vị trí sorry của skeleton
            vr = sch.verify(full_code)
            
            if not vr.get("pass") and not vr.get("sorries"):
                # Nếu vẫn fail thì mới báo lỗi
                errors = vr.get("errors", [])
                err_msg = errors[0].get("data", "Unknown error") if errors else "REPL Error"
                print(f"[{i+1}/{total}] {thm_name} ❌ Lỗi verify skeleton: {err_msg[:60]}...")
                continue

            # 4. Trích xuất thông tin từng sorry
            subgoals_data = []
            for s_item in vr.get("sorries", []):
                parsed = parse_proof_state(s_item.get("goal", ""), header=header)
                subgoals_data.append({
                    "pos": s_item.get("pos"),
                    "context": parsed.context,
                    "goal": parsed.goal
                })

            # 5. Cập nhật dữ liệu
            new_item = item.copy()
            new_item["subgoals"] = subgoals_data
            enriched_results.append(new_item)

            print(f"[{i+1}/{total}] {thm_name} ✅ Đã lấy {len(subgoals_data)} subgoals.")

        # Lưu kết quả dạng JSONL
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for res in enriched_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    finally:
        sch.close()
    
    print("\n" + "="*50)
    print(f"📊 HOÀN THÀNH!")
    print(f" - Tổng số mẫu xử lý thành công: {len(enriched_results)}")
    print(f" - Kết quả lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
