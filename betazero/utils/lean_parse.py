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
    current_line = ""

    for line in ctx_raw.splitlines():
        if not line:
            continue
        
        # In Lean goal output, hypotheses usually start at the beginning of the line (or 1-2 spaces).
        # Continuation lines for the same hypothesis are typically further indented.
        # Heuristic: if a line starts with substantial whitespace and doesn't look like a new hyp, join it.
        if (line.startswith("  ") or not (":" in line)) and ctx_lines:
            ctx_lines[-1] = f"{ctx_lines[-1]} {line.strip()}"
        else:
            line_strip = line.strip()
            if ":" in line_strip and not line_strip.startswith("case"):
                ctx_lines.append(line_strip)

    ctx = "\n".join(ctx_lines)
    return ProofState(context=ctx, goal=goal, header=header)

