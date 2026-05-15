import uuid
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
    extracted_code: str = ""  # parsed Lean code/body used for execution/stitching/logging
    children: tuple[ProofState, ...] = field(default_factory=tuple)
    prompt: str = ""  # exact prompt shown to the LLM for this content
    # THÊM ID DÙNG ĐỂ TRACE (KHÔNG THAM GIA VÀO HASH/EQ)
    id: str = field(
        default_factory=lambda: "A_" + uuid.uuid4().hex[:6], 
        compare=False, 
        hash=False
    )

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
