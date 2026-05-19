"""Lean example-wrapper helpers and verify message formatting."""

import re


def _shorten_feedback(text: str, *, max_chars: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def _message_position(msg: dict) -> str:
    """Return a compact Lean message position when the REPL provides one."""
    pos = msg.get("pos") or msg.get("startPos") or msg.get("endPos")
    if isinstance(pos, dict):
        line = pos.get("line")
        col = pos.get("column") or pos.get("col")
    else:
        line = msg.get("line")
        col = msg.get("column") or msg.get("col")

    if line is None:
        return ""
    if col is None:
        return f"L{line}"
    return f"L{line}:C{col}"


def _format_message(msg: dict, *, max_chars: int) -> str:
    data = _shorten_feedback(msg.get("data", ""), max_chars=max_chars)
    if not data:
        return ""
    pos = _message_position(msg)
    return f"{pos}: {data}" if pos else data


def format_lean_feedback(vr: dict, *, max_errors: int = 3, max_chars_per_error: int = 900) -> str:
    lines = [
        formatted
        for e in vr.get("errors", [])[:max_errors]
        for formatted in [_format_message(e, max_chars=max_chars_per_error)]
        if formatted
    ]
    if vr.get("system_errors"):
        lines.append(_shorten_feedback(str(vr["system_errors"]), max_chars=max_chars_per_error))
    return "\n".join(lines)

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
