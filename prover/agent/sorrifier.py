from __future__ import annotations

import json
import re
from typing import Any, Optional

from prover.agent.llm_client import LLMClient


class Sorrifier:
    """
    Hai chế độ:
    - fill_skeleton: thay/thu hẹp sorry (có thể kết hợp vị trí AST sau này).
    - fix_proof: nhận proof + lỗi REPL, sinh lại code.
    """

    _FILL_SYS = (
        "You edit Lean 4 code. Replace at most one `sorry` with a valid tactic sequence "
        "or a shorter proof sketch without new sorries if possible. Output JSON: "
        '{"code": "<full updated file>"}'
    )
    _FIX_SYS = (
        "You fix Lean 4 code given compiler errors. Output JSON: {\"code\": \"<full fixed file>\"}"
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @staticmethod
    def sorry_spans(code: str) -> list[tuple[int, int]]:
        """Vị trí ký tự (start, end) của từ `sorry` (heuristic, không thay thế parser Lean)."""
        spans: list[tuple[int, int]] = []
        for m in re.finditer(r"\bsorry\b", code):
            spans.append((m.start(), m.end()))
        return spans

    def fill_skeleton(self, skeleton_code: str, goal_hint: Optional[str] = None) -> str:
        hint = f"\nFocus goal context:\n{goal_hint}\n" if goal_hint else ""
        prompt = f"{hint}\n```lean\n{skeleton_code}\n```"
        raw = self.llm.complete(prompt, system=self._FILL_SYS, task="sorrifier_fill")
        parsed = self._parse_code_json(raw)
        if parsed:
            return parsed
        return self._replace_first_sorry_heuristic(skeleton_code, raw)

    def fix_proof(self, code: str, error_summary: str) -> str:
        prompt = f"Errors:\n{error_summary}\n\nCode:\n```lean\n{code}\n```"
        raw = self.llm.complete(prompt, system=self._FIX_SYS, task="sorrifier_fix")
        parsed = self._parse_code_json(raw)
        if parsed:
            return parsed
        return raw.strip() or code

    def _parse_code_json(self, text: str) -> str:
        text = text.strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return ""
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ""
        if not isinstance(obj, dict):
            return ""
        c = obj.get("code")
        return str(c).strip() if c else ""

    def _replace_first_sorry_heuristic(self, code: str, llm_raw: str) -> str:
        if "sorry" not in code:
            return code
        tactic = "trivial"
        fence = re.search(r"```lean\n([\s\S]*?)```", llm_raw)
        if fence:
            return fence.group(1).strip()
        if llm_raw.strip() and "sorry" not in llm_raw and "```" not in llm_raw:
            tactic = llm_raw.strip().splitlines()[0][:200]
        return re.sub(r"\bsorry\b", tactic, code, count=1)
