"""Lean example-wrapper helpers and verify message formatting."""


def extract_action_body(state_code: str) -> str:
    """Strip the `example ... := by` wrapper from compiled state code."""
    if ":= by\n" in state_code:
        body = state_code.split(":= by\n", 1)[1]
        return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
    return state_code


def format_lean_feedback(vr: dict) -> str:
    lines = [e.get("data", "") for e in vr.get("errors", [])[:12] if e.get("data", "")]
    if vr.get("system_errors"):
        lines.append(str(vr["system_errors"])[:800])
    return "\n".join(lines)

import re

def inject_patched_code_to_raw(raw_output: str, patched_full_code: str) -> str:
    """
    Tìm block ```lean4 ... ``` cuối cùng trong raw_output (giữ nguyên <think>) 
    và tráo ruột của nó bằng patched_full_code.
    """
    fallback_note = (
        "\nWait, my intended proof will be failed. I will fall back to using 'sorry' for the failed tactic.\n"
    )
    # Regex y hệt hàm get_lean_code của ông
    pattern = re.compile(r"(```lean4\s+)(.*?)(\s+```)", re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(raw_output))
    
    # Rủi ro: Model sinh thiếu tag ```lean4 (halucination nặng)
    if not matches:
        return raw_output.rstrip() + f"\n{fallback_note}\n```lean4\n{patched_full_code.strip()}\n```"
        
    last_match = matches[-1]
    fence_start = last_match.start(1)
    
    # Vị trí (start, end) của Group 2 - chính là phần ruột code Lean bị sai
    start_idx = last_match.start(2)
    end_idx = last_match.end(2)
    
    # Phẫu thuật thay ruột: Giữ phần đầu + Code mới + Giữ phần đuôi
    new_raw = (
        raw_output[:fence_start] + fallback_note + raw_output[fence_start:start_idx] +
        "\n" + patched_full_code.strip() + "\n" + 
        raw_output[end_idx:]
    )
    
    return new_raw
