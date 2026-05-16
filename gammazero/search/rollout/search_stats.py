from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gammazero.core import Action, ProofState


@dataclass
class StateStats:
    tactic_tries: int = 0
    skeleton_tries: int = 0

    tactic_probe_done: bool = False
    skeleton_probe_done: bool = False
    last_skeleton_new_children: int = 0
    bad_skeleton_rounds: int = 0
    skeleton_exhausted: bool = False

    exhausted: bool = False
    depth: int = 0

    last_score: float = 0.0
    active: bool = True

    best_tactic_r_env: float = 0.0
    incoming_skeleton_score: float = 0.0
    incoming_skeleton_r_env: float = 0.0
    parent_skeletons: list[tuple["ProofState", "Action"]] = field(default_factory=list)
