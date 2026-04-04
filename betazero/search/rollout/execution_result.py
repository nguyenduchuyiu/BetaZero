from __future__ import annotations

from dataclasses import dataclass, field

from betazero.core import ProofState


@dataclass(frozen=True)
class LeanExecutionResult:
    """Outcome of one Lean verify run: either a normal verify dict or a wrapped transport error."""

    state_code: str
    verify: dict
    subgoals: tuple[ProofState, ...] = field(default_factory=tuple)

    @classmethod
    def ok(cls, state_code: str, verify: dict, subgoals: list[ProofState]) -> LeanExecutionResult:
        return cls(state_code=state_code, verify=dict(verify), subgoals=tuple(subgoals))

    @classmethod
    def from_transport_error(cls, message: str, state_code: str = "") -> LeanExecutionResult:
        return cls(
            state_code=state_code,
            verify={
                "complete": False,
                "pass": False,
                "errors": [],
                "warnings": [],
                "sorries": [],
                "system_errors": message,
            },
            subgoals=(),
        )

    @property
    def system_errors(self) -> str:
        return str(self.verify.get("system_errors") or "").strip()

    @property
    def has_system_failure(self) -> bool:
        return bool(self.system_errors)
