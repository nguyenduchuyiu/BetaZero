import os
import random

from betazero.core.nodes import ProofState
from betazero.utils.lean_parse import parse_proof_state
from betazero.utils.lean_cmd import DEFAULT_OPEN


class TheoremDataset:
    def __init__(self, directory: str, *, scheduler):
        paths = sorted(f for f in os.listdir(directory) if f.endswith(".lean"))
        self.theorems: list[ProofState] = []
        for name in paths:
            p = os.path.join(directory, name)
            with open(p, encoding="utf-8") as f:
                content = f.read()
            vr = scheduler.verify(content)
            sorries = vr.get("sorries") or []
            goal_str = (sorries[0] or {}).get("goal", "") if sorries else ""
            ps = parse_proof_state(goal_str or "", header=DEFAULT_OPEN)
            if ps:
                self.theorems.append(ps)
        print(f"Loaded {len(self.theorems)} theorems from {directory}")

    def sample(self, n: int) -> list[ProofState]:
        return random.choices(self.theorems, k=n)
