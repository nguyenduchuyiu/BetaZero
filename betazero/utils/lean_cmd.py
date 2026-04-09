from __future__ import annotations

import re

from betazero.core.nodes import ProofState


DEFAULT_OPEN = "open BigOperators Nat Real Topology"


def sanitize_header(code: str) -> str:
    lines = code.splitlines()
    out = []
    in_header = True

    for line in lines:
        if in_header and line.strip().startswith("theorem"):
            in_header = False

        if in_header:
            # Catch ALL imports, not just Mathlib
            if re.match(r"^\s*import\s+", line):
                continue
            if re.match(r"^\s*set_option\s+maxHeartbeats\s+0\s*$", line):
                out.append("set_option maxHeartbeats 100000")
                continue

        out.append(line)

    return "\n".join(out)


def build_theorem(state: ProofState, code: str, *, name: str = "__bz_tmp") -> str:
    params = [
        f"({line.strip()})"
        for line in (state.context or "").splitlines()
        if line.strip() and ":" in line and not line.strip().startswith("case ")
    ]
    param_str = (" ".join(params) + " ") if params else ""

    indented = "\n".join(f"  {l}" for l in (code or "").strip().splitlines())

    header = state.header if state.header else DEFAULT_OPEN

    full_code = f"""{header}
theorem {name} {param_str}: {state.goal} := by
{indented}
"""

    # Sanitization is the verifier's responsibility (REPL/env compatibility).
    return full_code

