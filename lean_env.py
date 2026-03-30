import re

from nodes import ProofState
from ast_parser import get_lean_ast
from lean_verifier import Lean4ServerScheduler


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        candidate_code = self._build_cmd(state, code)
        vr = self.scheduler.verify(candidate_code)
        subgoals = [self._parse_proof_state(s.get("goal", "")) for s in vr.get("sorries", [])]
        return candidate_code, vr, subgoals

    def get_ast(self, code: str) -> list:
        return get_lean_ast(code)

    def dep_graph(self, proof_code: str) -> dict:
        """Classify 'have' hypotheses as core / benign / malignant."""
        result = self.scheduler.verify(proof_code)
        if result.get("errors") and not result.get("sorries"):
            return {"core": [], "benign": [], "malignant": []}
        sorry_lines = {s["pos"]["line"] for s in result.get("sorries", []) if s.get("pos")}
        code_lines = proof_code.splitlines()
        have_names = re.findall(r'\bhave\s+(\w+)\s*:', proof_code)
        malignant, core, benign = [], [], []
        for name in have_names:
            decl_idx = next(
                (i for i, l in enumerate(code_lines) if re.search(rf'\bhave\s+{re.escape(name)}\s*:', l)),
                -1,
            )
            if any(abs(sl - (decl_idx + 1)) <= 1 for sl in sorry_lines):
                malignant.append(name)
            elif re.search(rf'\b{re.escape(name)}\b', "\n".join(code_lines[decl_idx + 1:])):
                core.append(name)
            else:
                benign.append(name)
        return {"core": core, "benign": benign, "malignant": malignant}

    @staticmethod
    def _parse_proof_state(goal_str: str) -> ProofState:
        """Parse Lean Infoview goal string into a ProofState."""
        s = goal_str.strip()
        if "\n⊢ " in s:
            ctx, goal = s.rsplit("\n⊢ ", 1)
        elif s.startswith("⊢ "):
            ctx, goal = "", s[2:]
        else:
            ctx, goal = "", s
        return ProofState(context=ctx.strip(), goal=goal.strip())

    @staticmethod
    def _build_cmd(state: ProofState, code: str) -> str:
        """Wrap state context and goal into a compilable `example` block."""
        params = [
            f"({line.strip()})"
            for line in state.context.splitlines()
            if line.strip() and ":" in line and not line.strip().startswith("case ")
        ] if state.context else []
        param_str = (" ".join(params) + " ") if params else ""
        indented = "\n".join(f"  {l}" for l in code.strip().splitlines())
        return f"example {param_str}: {state.goal} := by\n{indented}"


if __name__ == "__main__":
    from lean_verifier import Lean4ServerScheduler
    env = LeanEnv(Lean4ServerScheduler(max_concurrent_requests=1, timeout=30, name="lean_env_test"))

    print("--- Testing execute() ---")
    initial_state = ProofState(context="n : Nat\nm : Nat", goal="n + m = m + n")
    _, _, children = env.execute(initial_state, "have h : 1 = 1 := by sorry\nsorry")
    print(f"Initial:\n{initial_state}\n")
    for i, child in enumerate(children, 1):
        print(f"Child {i}:\n{child}\n")

    print("--- Testing dep_graph() ---")
    deps = env.dep_graph(
        "have h_core : 1 = 1 := by exact rfl\n"
        "have h_benign : 2 = 2 := by exact rfl\n"
        "have h_malignant : 3 = 3 := by sorry\n"
        "exact h_core"
    )
    print(f" Core      : {deps['core']}")
    print(f" Benign    : {deps['benign']}")
    print(f" Malignant : {deps['malignant']}")
