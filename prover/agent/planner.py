from __future__ import annotations

import json
import re
from typing import Any

from prover.agent.llm_client import LLMClient
from prover.agent.proof_state import PlannerOutput


class Planner:
    """Sinh informal plan + formal skeleton (sorry) từ formal statement."""

    _SYSTEM = (
        "You are a formal proof planner for Lean 4. "
        "Reply with a single JSON object with keys: informal_plan (string), skeleton_code (string). "
        "skeleton_code must be valid Lean 4 with `sorry` placeholders where proofs are missing."
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def plan(self, formal_statement: str) -> PlannerOutput:
        prompt = f"Formal statement / context:\n```lean\n{formal_statement}\n```\nProduce JSON only."
        raw = self.llm.complete(prompt, system=self._SYSTEM, task="planner")
        data = self._parse_json_loose(raw)
        informal = str(data.get("informal_plan") or "").strip() or "[empty informal plan]"
        skeleton = str(data.get("skeleton_code") or "").strip()
        if not skeleton:
            skeleton = self._fallback_skeleton(formal_statement)
        return PlannerOutput(informal_plan=informal, skeleton_code=skeleton)

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

    def _fallback_skeleton(self, formal_statement: str) -> str:
        body = formal_statement.strip()
        if "theorem" not in body and "lemma" not in body:
            return (
                "import Mathlib\n\n"
                f"theorem user_stmt : True := by\n  sorry\n"
            )
        if ":=" in body:
            return body
        return body + " := by\n  sorry\n"
