"""Parse LLM proof output: CoT tags + ```lean4``` fence → lean snippet for LeanEnv."""

from __future__ import annotations

import re
import textwrap


_THINK_BLOCK = re.compile(r"(?is)<think>.*?</think>")
_THINK_CLOSE = re.compile(r"(?is)</think>")
_FENCE_LEAN4 = re.compile(r"```lean4(?:\s|$)(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_FENCE_LEAN = re.compile(r"```lean(?:\s|$)(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_FENCE_GENERIC = re.compile(r"```\s*(.*?)(?:```|$)", re.DOTALL)
_LEADING_BY = re.compile(r"(?is)^by(?:\s*\n|\s+)")
_LEADING_WRAPPER = re.compile(
    r"(?is)^(?:theorem|example)\b.*?:=\s*by(?:\s*\n|\s+)"
)


def get_lean_code(raw: str) -> str:
    """
    Trích xuất phần mã Lean cuối cùng từ khối ```lean4 hoặc ```lean trong output.
    Nếu không có, trả về toàn bộ chuỗi đã strip.
    """
    t = raw.strip()
    t = _THINK_BLOCK.sub("", t)
    t = _THINK_CLOSE.sub("", t).strip()

    matches = [
        *list(_FENCE_LEAN4.finditer(t)),
        *list(_FENCE_LEAN.finditer(t)),
    ]
    if matches:
        code = matches[-1].group(1).strip()
    else:
        generic = _FENCE_GENERIC.search(t)
        code = generic.group(1).strip() if generic else t

    had_wrapper = _LEADING_WRAPPER.match(code) is not None
    code = _LEADING_WRAPPER.sub("", code, count=1)
    if had_wrapper:
        code = textwrap.dedent(code)
    code = _LEADING_BY.sub("", code, count=1)
    return code.strip()
