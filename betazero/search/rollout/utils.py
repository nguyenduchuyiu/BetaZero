"""Lean example-wrapper helpers and verify message formatting."""


def extract_action_body(state_code: str) -> str:
    """Strip the `example ... := by` wrapper from compiled state code."""
    if ":= by\n" in state_code:
        body = state_code.split(":= by\n", 1)[1]
        return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
    return state_code


def format_lean_feedback(vr: dict) -> str:
    lines = [e.get("data", "") for e in vr.get("errors", [])[:12] if e.get("data", "")]
    if vr.get("system_errors"):
        lines.append(str(vr["system_errors"])[:800])
    return "\n".join(lines)
