"""Extract RAG queries from sorry positions in Lean code via REPL."""

from prover.lean.verifier import verify_lean4_file


def parse_proof_state(state: str) -> dict:
    if '⊢' not in state:
        return {"hypotheses": [], "goal": state.strip()}
    given, goal = state.rsplit('⊢', 1)
    hyps = []
    for line in given.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isspace() and hyps:
            hyps[-1] += " " + line  # continuation
        else:
            parts = line.split(' : ', 1)
            if len(parts) == 2:
                hyps.append({"name": parts[0].strip(), "type": parts[1].strip()})
            else:
                hyps.append({"name": None, "type": line})
    return {"hypotheses": hyps, "goal": goal.strip()}


def extract_queries(code: str, timeout=120) -> list[dict]:
    """Send code to REPL, return structured proof state at each sorry.

    Returns list of:
      {"line": int, "col": int, "raw": str,
       "hypotheses": [{"name":..., "type":...}], "goal": str}
    """
    result = verify_lean4_file(code, timeout=timeout)
    queries = []
    for s in result.get("sorries", []):
        parsed = parse_proof_state(s["goal"])
        queries.append({
            "line": s["pos"]["line"],
            "col": s["pos"]["column"],
            "raw": s["goal"],
            **parsed,
        })
    return queries


# --- test ---
if __name__ == "__main__":
    code = open("tet.lean").read()
    print(f"=== Input ===\n{code}")

    queries = extract_queries(code)
    print(f"Found {len(queries)} sorry(s)\n")

    for i, q in enumerate(queries, 1):
        print(f"--- sorry #{i} (line {q['line']}) ---")
        print(f"RAG query:\n{q['raw']}\n")
