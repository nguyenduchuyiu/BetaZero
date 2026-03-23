from __future__ import annotations

import re

from prover.agent.llm_client import LLMClient


class Solver:
    """Sinh whole formal proof (không sorry) cho một subgoal."""

    _SYSTEM = (
        "You are a Lean 4 prover. Output a complete Lean file that discharges the goal without `sorry`. "
        "Prefer `import Mathlib` if needed."
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def solve(self, formal_subgoal: str, prior_error: str | None = None) -> str:
        err = f"\nPrevious error / feedback:\n{prior_error}\n" if prior_error else ""
        prompt = f"{err}\nProve:\n```lean\n{formal_subgoal}\n```\nOutput only Lean code."
        raw = self.llm.complete(prompt, system=self._SYSTEM, task="solver")
        code = self._extract_lean_fence(raw)
        return code.strip() or raw.strip()

    @staticmethod
    def _extract_lean_fence(text: str) -> str:
        m = re.search(r"```(?:lean)?\n([\s\S]*?)```", text)
        if m:
            return m.group(1)
        return text
