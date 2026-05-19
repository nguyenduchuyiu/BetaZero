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
    scaffold_code: str = ""  # full Lean scaffold containing this state's target `sorry`
    target_index: int = 0  # textual `sorry` index inside scaffold_code
    target_kind: str = ""  # root, skeleton_child, mini_skeleton_child, ...
    parent_action_id: str = field(default="", compare=False, hash=False)

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
    verify_code: str = field(default="", compare=False, hash=False)
    stitched_code: str = field(default="", compare=False, hash=False)
    patched_code: str = field(default="", compare=False, hash=False)
    lean_feedback: str = field(default="", compare=False, hash=False)
    target_child_index: int | None = field(default=None, compare=False, hash=False)
    # THÊM ID DÙNG ĐỂ TRACE (KHÔNG THAM GIA VÀO HASH/EQ)
    id: str = field(
        default_factory=lambda: "A_" + uuid.uuid4().hex[:6], 
        compare=False, 
        hash=False
    )

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
