#!/usr/bin/env python3
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from betazero.policy.output_parser import get_lean_code


INPUT_FILE = Path("conversations.json")
OUT_FILE = Path("automation-output/SFT_data.json")


@dataclass
class ProcessedSample:
    id: str
    theorem: str
    prompt: str
    raw_output: str
    extracted_code: str
    error_reason: str = ""


def main():
    if not INPUT_FILE.exists():
        print(f"Khong tim thay file dau vao {INPUT_FILE}")
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sft_data = []
    print(f"Dang convert {len(data)} conversations...")

    for i, conv in enumerate(data, start=1):
        mapping = conv.get("mapping", {})
        prompt_content = ""
        response_content = ""

        for v in mapping.values():
            msg = v.get("message")
            if not msg:
                continue
            frags = msg.get("fragments") or []
            content = "".join(f.get("content", "") for f in frags if isinstance(f, dict))
            if not content:
                content = "".join(f for f in frags if isinstance(f, str))

            if "[PROBLEM]" in content:
                prompt_content = content
            elif "<think>" in content:
                response_content = content

        if not prompt_content or not response_content:
            continue

        m = re.search(r"```lean4\n(.*?)\n```", prompt_content, re.DOTALL)
        if not m:
            continue

        sample = ProcessedSample(
            id=f"my_theorem_{i}",
            theorem=m.group(1),
            prompt=prompt_content,
            raw_output=response_content,
            extracted_code=get_lean_code(response_content),
        )
        sft_data.append(asdict(sample))

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(sft_data)} samples to {OUT_FILE}")


if __name__ == "__main__":
    main()