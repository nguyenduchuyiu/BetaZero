from __future__ import annotations

from typing import Optional

from prover.agent.lean_repl import LeanRepl, LeanReplResult


class ProofVerifier:
    """Gửi proof lên REPL; Pass/Fail + errors cho ErrorHandler / Sorrifier."""

    def __init__(self, repl: LeanRepl | None = None, timeout: int = 300) -> None:
        self.repl = repl or LeanRepl(timeout=timeout)

    def verify(self, code: str, timeout: Optional[int] = None) -> LeanReplResult:
        return self.repl.eval(code, timeout=timeout)
