from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateStats:
    tactic_tries: int = 0
    skeleton_tries: int = 0

    tactic_probe_done: bool = False
    skeleton_probe_done: bool = False

    exhausted: bool = False
    depth: int = 0

    last_score: float = 0.0
    active: bool = True
