from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from prover.lean.verifier import verify_lean4_file


@dataclass
class LeanReplResult:
    """Kết quả chuẩn hóa từ Lean REPL (mọi layer gọi qua đây)."""

    pass_ok: bool
    complete: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    infos: list[dict[str, Any]]
    sorries: list[dict[str, Any]]
    system_errors: str
    verified_code: str
    verify_time: float
    raw: dict[str, Any]


class LeanRepl:
    """
    Wrapper duy nhất cho Lean 4: gửi code, parse Pass/Fail + message.
    Không spawn process pool — an toàn khi import từ worker/multiprocessing.
    """

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout

    def eval(self, code: str, timeout: Optional[int] = None) -> LeanReplResult:
        t = self.timeout if timeout is None else timeout
        raw = verify_lean4_file(code, timeout=t)
        return LeanReplResult(
            pass_ok=bool(raw.get("pass")),
            complete=bool(raw.get("complete")),
            errors=list(raw.get("errors") or []),
            warnings=list(raw.get("warnings") or []),
            infos=list(raw.get("infos") or []),
            sorries=list(raw.get("sorries") or []),
            system_errors=str(raw.get("system_errors") or ""),
            verified_code=str(raw.get("verified_code") or code),
            verify_time=float(raw.get("verify_time") or 0.0),
            raw=raw,
        )
