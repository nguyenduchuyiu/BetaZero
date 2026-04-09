import argparse
import json
import re
from pathlib import Path


def fix_text(s: str) -> str:
    if not s:
        return s
    return s.replace(":= by\n  sorry", ":= sorry")

def extract_code_from_raw(raw: str) -> str:
    """
    Extract the last fenced code block content from raw_output.
    Accepts ```lean4 ...``` or ```lean ...``` or ```...```.
    Returns "" if not found.
    """
    if not raw:
        return ""
    # find last fenced block
    start = raw.rfind("```")
    if start == -1:
        return ""
    end = raw.find("```", start + 3)
    if end == -1:
        return ""
    block = raw[start + 3 : end]
    block = block.lstrip("\n")
    # drop optional language tag on first line
    first_nl = block.find("\n")
    if first_nl != -1:
        first_line = block[:first_nl].strip()
        if first_line in {"lean", "lean4"} or re.fullmatch(r"[A-Za-z0-9_+-]+", first_line):
            block = block[first_nl + 1 :]
    return block.rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to skeleton_samples.json")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input JSON")
    ap.add_argument("--no-raw", action="store_true", help="Do not touch raw_output")
    ap.add_argument("--no-extract", action="store_true", help="Do not re-extract extracted_code from raw_output")
    args = ap.parse_args()

    p = Path(args.json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    results = data.get("results", [])

    n_raw_changed = 0
    n_extracted_changed = 0
    for item in results:
        if not args.no_raw:
            rb = item.get("raw_output", "")
            ra = fix_text(rb)
            if ra != rb:
                item["raw_output"] = ra
                n_raw_changed += 1

        if not args.no_extract:
            raw_now = item.get("raw_output", "")
            extracted = extract_code_from_raw(raw_now)
            if extracted and extracted != item.get("extracted_code", ""):
                item["extracted_code"] = extracted
                n_extracted_changed += 1

    out_path = p if args.in_place else p.with_suffix(".fixed.json")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"changed_raw_output={n_raw_changed}")
    print(f"changed_extracted_code={n_extracted_changed}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

