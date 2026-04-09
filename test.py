import json
import re
from pathlib import Path
from collections import Counter

def check_skeleton_quality(file_path: str):
    print(f"🔍 Đang nội soi file: {file_path}...\n")
    
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results: return print("❌ Data rỗng!")

    stats = Counter()
    bad_cases = []
    tokens = {"prompt": [], "raw": [], "code": []}

    # FIX LỖI MÂU THUẪN: Đã cho phép ở Regex thì phải gỡ khỏi Forbidden!
    FORBIDDEN_TACTICS = ["simp", "nlinarith", "omega", "aesop"]
    ALLOWED_HAVE = re.compile(r":=\s*(by\s+)?(sorry|rfl|trivial|ring|linarith|norm_num|[a-zA-Z0-9_.]+|⟨.*?⟩)\s*$")

    for idx, item in enumerate(results):
        raw, code = item.get("raw_output", ""), item.get("extracted_code", "")
        
        # Gom token stats
        tokens["prompt"].append(len(item.get("prompt", "")) // 4)
        tokens["raw"].append(len(raw) // 4)
        tokens["code"].append(len(code) // 4)

        errs = set()
        
        # 1. Check độ dài & lải nhải
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match and len(think_match.group(1)) // 4 > 1500:
            errs.add("think_too_long")
        if raw.rfind("```") != -1 and len(raw[raw.rfind("```") + 3:].strip()) > 5:
            errs.add("yapping_after_codeblock")

        # 2. Check từng dòng code
        for line in (code or "").splitlines():
            line = line.strip()
            if not line: continue
            
            if line.startswith("have "):
                if ":" not in line: errs.add("have_missing_type")
                if not ALLOWED_HAVE.search(line.split("--")[0].strip()): errs.add("have_invalid_ending")
            
            for tac in FORBIDDEN_TACTICS:
                if re.search(rf"\b{tac}\b", line): errs.add(f"forbidden_{tac}")

        # 3. Tổng hợp
        if not errs:
            stats["perfect"] += 1
        else:
            for e in errs: stats[e] += 1
            # Chỉ ném vào bad_cases nếu có lỗi khác ngoài "think_too_long"
            if errs - {"think_too_long"}:
                bad_cases.append({"path": item.get("path", f"idx={idx}"), "code": code, "errs": list(errs)})

    # --- IN BÁO CÁO ---
    print("="*50)
    print(f"📊 BÁO CÁO (Tổng: {len(results)} | Hoàn hảo: {stats['perfect']} - {stats['perfect']/len(results):.1%})")
    print("="*50)
    for k, v in stats.items():
        if k != "perfect": print(f" - {k:<25}: {v} ca")

    # --- GHI FILE MARKDOWN ---
    out_dir = Path(file_path).parent
    if bad_cases:
        md_path = out_dir / "violations.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Violations Report\n\n")
            for bc in bad_cases:
                f.write(f"## {bc['path']}\n**Errors**: {', '.join(bc['errs'])}\n```lean4\n{(bc['code'] or '').strip()}\n```\n\n")
        print(f"\n🧾 Đã lưu file vi phạm: {md_path}")

    # --- TOKEN STATS ---
    print("\n📦 TOKEN STATS (approx = len/4):")
    for k, arr in tokens.items():
        arr.sort()
        print(f" - {k:<10}: min={arr[0]}, p50={arr[len(arr)//2]}, p90={arr[int(0.9*len(arr))]}, max={arr[-1]}" if arr else f" - {k:<10}: n=0")

    # --- PLOT VẼ NHANH GỌN ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        for k, alpha in [("raw", 0.6), ("prompt", 0.4), ("code", 0.5)]:
            plt.hist(tokens[k], bins=30, alpha=alpha, label=k)
        plt.title("Token distribution"), plt.legend(), plt.tight_layout()
        plt.savefig(out_dir / "token_distribution.png", dpi=160); plt.close()
        print(f"🖼️ Đã lưu plot.")
    except ImportError:
        pass

if __name__ == "__main__":
    check_skeleton_quality("automation-output/skeleton_samples.json")