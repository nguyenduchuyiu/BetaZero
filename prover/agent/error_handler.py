from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from prover.agent.lean_repl import LeanReplResult

T = TypeVar("T")


@dataclass
class ParsedReplError:
    severity: str
    summary: str
    raw: dict[str, Any]


class ErrorHandler:
    """
    Parse lỗi từ REPL, quyết định có nên thử sửa hay escalate.
    """

    _UNFIXABLE = re.compile(
        r"unknown identifier|invalid field notation|cannot synthesize|type mismatch.*sorry",
        re.IGNORECASE,
    )

    def summarize(self, result: LeanReplResult) -> str:
        parts: list[str] = []
        if result.system_errors:
            parts.append(result.system_errors.strip())
        for e in result.errors:
            parts.append(str(e.get("data") or e))
        return "\n".join(parts).strip()

    def parse_errors(self, result: LeanReplResult) -> list[ParsedReplError]:
        out: list[ParsedReplError] = []
        for e in result.errors:
            out.append(
                ParsedReplError(
                    severity="error",
                    summary=str(e.get("data") or e),
                    raw=e,
                )
            )
        return out

    def can_fix(self, result: LeanReplResult) -> bool:
        if result.pass_ok and result.complete:
            return False
        if not result.errors and not result.system_errors:
            return True
        blob = self.summarize(result)
        if not blob:
            return True
        return self._UNFIXABLE.search(blob) is None

    def retry_loop(
        self,
        *,
        max_attempts: int,
        initial: T,
        step: Callable[[T, int], T],
        verify: Callable[[T], LeanReplResult],
        should_stop: Callable[[LeanReplResult], bool],
    ) -> tuple[T, LeanReplResult, bool]:
        """
        Vòng lặp retry chung: `step` sửa artifact, `verify` gọi REPL, `should_stop` khi thành công.
        Trả về (artifact, kết quả REPL cuối, success).
        """
        cur: T = initial
        last = verify(cur)
        if should_stop(last):
            return cur, last, True
        for attempt in range(1, max_attempts + 1):
            if not self.can_fix(last):
                return cur, last, False
            cur = step(cur, attempt)
            last = verify(cur)
            if should_stop(last):
                return cur, last, True
        return cur, last, False
