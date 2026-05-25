from __future__ import annotations

from gammazero.core.nodes import ProofState


def extract_proof_body(state_code: str) -> str:
    """Strip the `example/theorem ... := by` wrapper from compiled Lean code."""
    if ":= by\n" in state_code:
        body = state_code.split(":= by\n", 1)[1]
        return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
    return state_code


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
        
        # In Lean's goal output, hypotheses usually start at column 0 (or 1-2
        # spaces). Continuation lines for the same hypothesis are typically
        # indented further. Heuristic: if a line is heavily indented and does
        # not look like a new hypothesis, append it to the previous one.
        if (line.startswith("  ") or not (":" in line)) and ctx_lines:
            ctx_lines[-1] = f"{ctx_lines[-1]} {line.strip()}"
        else:
            line_strip = line.strip()
            if ":" in line_strip and not line_strip.startswith("case"):
                ctx_lines.append(line_strip)

    ctx = "\n".join(ctx_lines)
    return ProofState(context=ctx, goal=goal, header=header)
