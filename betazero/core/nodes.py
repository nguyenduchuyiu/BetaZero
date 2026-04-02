from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


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
    """AND-node: a tactic sequence or proof skeleton applied to a ProofState."""
    action_type: Literal["tactic", "skeleton"]
    content: str
    children: tuple[ProofState, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
