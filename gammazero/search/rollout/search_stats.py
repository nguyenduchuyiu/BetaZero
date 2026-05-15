from __future__ import annotations

from dataclasses import dataclass


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
