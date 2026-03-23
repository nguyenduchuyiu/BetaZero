from __future__ import annotations

import re

from prover.agent.llm_client import LLMClient
from prover.agent.proof_state import RouteDecision


class Router:
    """Quyết định DECOMPOSE vs SOLVE cho một subgoal."""

    _SYSTEM = (
        "You route formal proof subgoals. Reply with exactly one token: DECOMPOSE or SOLVE. "
        "DECOMPOSE if the goal should be split into smaller lemmas. SOLVE if a single proof is appropriate."
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def route(self, formal_subgoal: str) -> RouteDecision:
        prompt = f"Subgoal:\n```lean\n{formal_subgoal}\n```"
        raw = self.llm.complete(prompt, system=self._SYSTEM, task="router").strip().upper()
        m = re.search(r"\b(DECOMPOSE|SOLVE)\b", raw)
        if m and m.group(1) == "DECOMPOSE":
            return RouteDecision.DECOMPOSE
        return RouteDecision.SOLVE
