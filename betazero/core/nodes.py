from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

NodeStatus = Literal["OPEN", "SOLVED", "FAILED"]


@dataclass(frozen=True)
class ProofState:
    """OR-node: a proof state (context, goal) in the AND/OR search graph."""
    context: str
    goal: str
    header: str = ""  # import lines from the source .lean file

    def __str__(self) -> str:
        return f"{self.context}\n⊢ {self.goal}" if self.context else f"⊢ {self.goal}"


@dataclass(frozen=True)
class Action:
    """AND-node: tactic or skeleton. `content` is raw LLM output; Lean execution uses extracted ```lean4``` body."""
    action_type: Literal["tactic", "skeleton"]
    content: str
    children: tuple[ProofState, ...] = field(default_factory=tuple)
    prompt: str = ""  # exact prompt shown to the LLM for this content
    is_sc_tactic: bool = False  # phase-2 self-correct rollout sample (training split)

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
