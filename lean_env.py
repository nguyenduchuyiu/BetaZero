"""
Lean Environment Adapter.

Stateless adapter that bridges domain objects (ProofState) and the Lean verifier.
"""

import re
from typing import List

from nodes import ProofState
from ast_parser import get_lean_ast
from lean_verifier import verify_lean_code

class LeanEnv:
    """Wrapper for Lean verification and state parsing."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """
        Đóng gói tactic code, chạy Lean và trả về bộ 3 quyền lực:
        (full_cmd_đã_build, verify_result_gốc, danh_sách_subgoals_nếu_có)
        """
        candidate_code = self._build_cmd(state, code)
        vr = verify_lean_code(candidate_code, timeout=self.timeout)
        subgoals = [self._parse_proof_state(s.get("goal", "")) for s in vr.get("sorries", [])]
        return candidate_code, vr, subgoals

    def get_ast(self, code: str) -> list:
        """Parses the Lean code string into an Abstract Syntax Tree (AST)."""
        return get_lean_ast(code)

    def dep_graph(self, proof_code: str) -> dict:
        """
        Analyzes the dependency graph of 'have' hypotheses in the proof code.
        Classifies hypotheses into three categories:
        - malignant: Unprovable (body is or contains 'sorry').
        - core: Proven and actively referenced in subsequent steps.
        - benign: Proven but never referenced again.
        """
        result = verify_lean_code(proof_code, timeout=self.timeout)
        
        if result.get("errors") and not result.get("sorries"):
            return {"core": [], "benign": [], "malignant": []}

        # Extract line numbers where 'sorry' appears
        sorry_lines = {s["pos"]["line"] for s in result.get("sorries", []) if s.get("pos")}
        code_lines = proof_code.splitlines()
        
        # Find all declared hypothesis names using 'have <name> :'
        have_names = re.findall(r'\bhave\s+(\w+)\s*:', proof_code)

        malignant, core, benign = [], [], []
        for name in have_names:
            # Find the line index where the hypothesis is declared
            decl_idx = next(
                (i for i, l in enumerate(code_lines) if re.search(rf'\bhave\s+{re.escape(name)}\s*:', l)),
                -1,
            )
            
            # If a 'sorry' is within 1 line of the declaration, it is unprovable
            if any(abs(sl - (decl_idx + 1)) <= 1 for sl in sorry_lines):
                malignant.append(name)
            # If the name is used in the remaining code, it is a core dependency
            elif re.search(rf'\b{re.escape(name)}\b', "\n".join(code_lines[decl_idx + 1:])):
                core.append(name)
            # Otherwise, it is valid but unused
            else:
                benign.append(name)

        return {"core": core, "benign": benign, "malignant": malignant}

    @staticmethod
    def _parse_proof_state(goal_str: str) -> ProofState:
        """Parses Lean's raw Infoview goal string into a ProofState object."""
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
        """Wraps the context and goal into a compilable 'example' block."""
        params = [
            f"({line.strip()})"
            for line in state.context.splitlines()
            if line.strip() and ":" in line and not line.strip().startswith("case ")
        ] if state.context else []
        
        param_str = (" ".join(params) + " ") if params else ""
        indented = "\n".join(f"  {l}" for l in code.strip().splitlines())
        
        return f"example {param_str}: {state.goal} := by\n{indented}"

if __name__ == "__main__":
    # Initialize the environment
    env = LeanEnv(timeout=30)

    # 1. Test execute()
    print("--- Testing execute() ---")
    initial_state = ProofState(context="n : Nat\nm : Nat", goal="n + m = m + n")
    
    # Using 'sorry' should return 1 unsolved subgoal
    tactic_code = '''   have h : 1 = 1 := by sorry
sorry'''
    children = env.execute(initial_state, tactic_code)
    
    print(f"Initial:\n{initial_state}\n")
    print(f"Generated {len(children)} child state(s) after '{tactic_code}':")
    for i, child in enumerate(children, 1):
        print(f"Child {i}:\n{child}\n")

    # 2. Test dep_graph()
    print("--- Testing dep_graph() ---")
    # h_core is used, h_benign is unused, h_malignant is unprovable (sorry)
    proof_snippet = """have h_core : 1 = 1 := by exact rfl
have h_benign : 2 = 2 := by exact rfl
have h_malignant : 3 = 3 := by sorry
exact h_core"""
    
    deps = env.dep_graph(proof_snippet)
    print("Dependency Graph:")
    print(f" Core      : {deps['core']}")
    print(f" Benign    : {deps['benign']}")
    print(f" Malignant : {deps['malignant']}")