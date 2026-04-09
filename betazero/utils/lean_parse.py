from __future__ import annotations

from betazero.core.nodes import ProofState


def parse_proof_state(goal_str: str, *, header: str = "") -> ProofState:
    """Parse Lean Infoview goal string (with `⊢`) into a ProofState."""
    s = (goal_str or "").strip()
    if not s or "Goals accomplished" in s or "no goals" in s:
        return ProofState(context="", goal="SOLVED_OR_EMPTY", header=header)

    parts = s.split("⊢")
    if len(parts) > 1:
        ctx_raw = parts[0].strip()
        main_goal_raw = parts[1].strip()

        goal_lines: list[str] = []
        for line in main_goal_raw.splitlines():
            if line.startswith("case ") or "Goals accomplished" in line:
                break
            goal_lines.append(line)
        goal = "\n".join(goal_lines).strip()
    else:
        ctx_raw, goal = "", s

    if goal.lower() == "sorry":
        goal = "ELABORATION_FAULT"

    ctx_lines: list[str] = []
    for line in ctx_raw.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("case"):
            ctx_lines.append(line)
    ctx = "\n".join(ctx_lines)
    return ProofState(context=ctx, goal=goal, header=header)

