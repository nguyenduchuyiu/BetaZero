import os
import re
import random

from betazero.data.nodes import ProofState


def _extract_params(params_str: str) -> list[str]:
    """Extract inner content of each top-level bracket group."""
    groups, depth, start = [], 0, -1
    for i, c in enumerate(params_str):
        if c in "([{" and depth == 0:
            depth, start = 1, i + 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and start != -1:
                inner = params_str[start:i].strip()
                if inner:
                    groups.append(inner)
                start = -1
    return groups


def parse_lean_file(path: str) -> ProofState | None:
    """Parse a miniF2F .lean file into a ProofState."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'\btheorem\s+\w+(.*?):=\s*by\s+sorry', content, re.DOTALL)
    if not m:
        return None
    sig = m.group(1).strip()
    depth, colon_pos = 0, -1
    for i, c in enumerate(sig):
        if c in "([{":   depth += 1
        elif c in ")]}": depth -= 1
        elif c == ":" and depth == 0 and (i + 1 >= len(sig) or sig[i + 1] != "="):
            colon_pos = i
    if colon_pos == -1:
        return None
    goal    = sig[colon_pos + 1:].strip()
    context = "\n".join(_extract_params(sig[:colon_pos]))
    return ProofState(context=context, goal=goal)


class TheoremDataset:
    def __init__(self, directory: str):
        paths = sorted(f for f in os.listdir(directory) if f.endswith(".lean"))
        self.theorems: list[ProofState] = []
        for name in paths:
            ps = parse_lean_file(os.path.join(directory, name))
            if ps:
                self.theorems.append(ps)
        print(f"Loaded {len(self.theorems)} theorems from {directory}")

    def sample(self, n: int) -> list[ProofState]:
        return random.choices(self.theorems, k=n)
