"""Parse LLM proof output: CoT tags + ```lean4``` fence → lean snippet for LeanEnv."""

from __future__ import annotations

def get_lean_code(raw: str) -> str:
    """
    Trích xuất phần mã Lean cuối cùng từ khối ```lean4 hoặc ```lean trong output.
    Nếu không có, trả về toàn bộ chuỗi đã strip.
    """
    import re
    t = raw.strip()
    matches = list(re.finditer(r"```lean4(.*?)(?:```|$)|```lean(.*?)(?:```|$)", t, re.DOTALL | re.IGNORECASE))
    if matches:
        last = matches[-1]
        code = last.group(1) if last.group(1) is not None else last.group(2)
        return code
    return t