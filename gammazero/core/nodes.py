import uuid
from dataclasses import dataclass, field
from typing import Literal

NodeStatus = Literal["OPEN", "SOLVED", "FAILED", "RESERVED"]


@dataclass(frozen=True)
class ProofState:
    """OR-node: a (context, goal) proof state in the AND/OR search graph."""
    context: str
    goal: str
    header: str = ""  # imports from the source .lean file
    scaffold_code: str = ""  # full Lean scaffold containing this state's target `sorry`
    target_index: int = 0  # textual index of the target `sorry` in scaffold_code
    target_kind: str = ""  # e.g. root, skeleton_child, mini_skeleton_child
    parent_action_id: str = field(default="", compare=False, hash=False)

    def __str__(self) -> str:
        return f"{self.context}\n⊢ {self.goal}" if self.context else f"⊢ {self.goal}"


@dataclass(frozen=True)
class Action:
    """AND-node: a tactic or skeleton. `content` holds the raw LLM output; Lean uses the extracted ```lean4``` body."""
    action_type: Literal["tactic", "skeleton"]
    content: str
    extracted_code: str = ""  # parsed Lean code used for execution, stitching, and logging
    children: tuple[ProofState, ...] = field(default_factory=tuple)
    prompt: str = ""  # exact prompt sent to the LLM
    verify_code: str = field(default="", compare=False, hash=False)
    stitched_code: str = field(default="", compare=False, hash=False)
    patched_code: str = field(default="", compare=False, hash=False)
    lean_feedback: str = field(default="", compare=False, hash=False)
    target_child_index: int | None = field(default=None, compare=False, hash=False)
    # Trace ID; excluded from hash/equality.
    id: str = field(
        default_factory=lambda: "A_" + uuid.uuid4().hex[:6],
        compare=False,
        hash=False,
    )

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
