from __future__ import annotations

import json
import re
from typing import Any

from prover.agent.llm_client import LLMClient


class Decomposer:
    """Chia subgoal thành danh sách subgoal nhỏ hơn (đưa vào queue orchestrator)."""

    _SYSTEM = (
        "You decompose a Lean proof goal. Reply with JSON only: "
        '{"subgoals": ["<formal statement 1>", "<formal statement 2>", ...]}. '
        "Each item should be a standalone Lean-friendly statement string."
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def decompose(self, formal_subgoal: str) -> list[str]:
        prompt = f"Goal to decompose:\n```lean\n{formal_subgoal}\n```"
        raw = self.llm.complete(prompt, system=self._SYSTEM, task="decompose")
        data = self._parse_json_loose(raw)
        subs = data.get("subgoals")
        if isinstance(subs, list):
            out = [str(x).strip() for x in subs if str(x).strip()]
            if out:
                return out
        return self._fallback(formal_subgoal)

    def _parse_json_loose(self, text: str) -> dict[str, Any]:
        text = text.strip()
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            text = m.group(0)
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _fallback(self, formal_subgoal: str) -> list[str]:
        return [
            formal_subgoal + "\n-- split part 1",
            formal_subgoal + "\n-- split part 2",
        ]
