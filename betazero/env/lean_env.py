import re

from betazero.core import ProofState
from betazero.env.ast_parser import get_lean_ast
from betazero.env import Lean4ServerScheduler
from betazero.search.sorrifier.dependency_analyzer import ExprDependencyAnalyzer
from betazero.env.expr_parser import get_lean_expr_tree


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler
        self.analyzer = ExprDependencyAnalyzer()

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        candidate_code = self._build_cmd(state, code)
        vr = self.scheduler.verify(candidate_code)
        subgoals = [
            self._parse_proof_state(s.get("goal", ""), header=state.header)
            for s in vr.get("sorries", [])
        ]
        return candidate_code, vr, subgoals

    def get_ast(self, code: str) -> list:
        return get_lean_ast(code)

    def analyze_dependencies(self, proof_code: str) -> dict:
        """
        Classify subgoals using Lean 4 Expr Tree deep analysis.
        Returns classifications for: core_solved, core_failed, malignant, benign.
        """
        
        # We don't even need scheduler.verify here because Expr Tree compilation 
        # naturally fails or succeeds, exposing the core structures.
        ast_expr_list = get_lean_expr_tree(proof_code)
        
        empty_classification = {
            "core_solved": [], "core_failed": [], "malignant": [], "benign": []
        }
        
        if not ast_expr_list:
            return empty_classification
            
        # The target theorem is usually the last block extracted
        root_expr = ast_expr_list[-1].get("expr_tree", {})
        classification = self.analyzer.classify_skeleton_subgoals(root_expr)
        
        return classification

    @staticmethod
    def _parse_proof_state(goal_str: str, header: str = "") -> ProofState:
        """Parse Lean Infoview goal string into a ProofState (with inherited header)."""
        s = goal_str.strip()
        if "\n⊢ " in s:
            ctx, goal = s.rsplit("\n⊢ ", 1)
        elif s.startswith("⊢ "):
            ctx, goal = "", s[2:]
        else:
            ctx, goal = "", s
        return ProofState(context=ctx.strip(), goal=goal.strip(), header=header)

    @staticmethod
    def _sanitize_header(header: str) -> str:
        """Remove redundant Mathlib imports and normalize maxHeartbeats for Lean execution."""
        if not header:
            return ""

        out: list[str] = []
        for line in header.splitlines():
            if re.match(r'^\s*import\s+Mathlib', line):
                continue
            if re.match(r'^\s*set_option\s+maxHeartbeats\s+0\s*$', line):
                out.append("set_option maxHeartbeats 100000")
                continue
            out.append(line)
        return "\n".join(out).strip()

    @staticmethod
    def _build_cmd(state: ProofState, code: str) -> str:
        """Wrap state header, context and goal into a compilable `example` block."""
        params = [
            f"({line.strip()})"
            for line in state.context.splitlines()
            if line.strip() and ":" in line and not line.strip().startswith("case ")
        ] if state.context else []

        param_str = (" ".join(params) + " ") if params else ""
        indented = "\n".join(f"  {l}" for l in code.strip().splitlines())
        prefix_header = LeanEnv._sanitize_header(state.header) if state.header else ""
        prefix = (prefix_header + "\n\n") if prefix_header else ""
        return f"{prefix}example {param_str}: {state.goal} := by\n{indented}"
