import re

from betazero.core import ProofState
from betazero.env.ast_parser import get_lean_ast
from betazero.env import Lean4ServerScheduler
from betazero.env.expr_parser import get_lean_expr_tree


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
        
        subgoals = []
        if vr.get("pass"):
            for s in vr.get("sorries", []):
                ps = self._parse_proof_state(s.get("goal", ""), header=state.header)
                if ps.goal not in ["SOLVED_OR_EMPTY", "ELABORATION_FAULT"]:
                    subgoals.append(ps)

        return candidate_code, vr, subgoals

    def get_ast(self, code: str) -> list:
        return get_lean_ast(code)

    def analyze_dependencies(self, proof_code: str) -> dict:
        """
        Classify subgoals using Lean 4 Expr Tree deep analysis.
        Returns classifications for: core_solved, core_failed, malignant, benign.
        """
        from betazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
        ast_expr_list = get_lean_expr_tree(proof_code)
        
        empty_classification = {
            "core_solved": [], "core_failed": [], "malignant": [], "benign": []
        }
        
        if not ast_expr_list:
            return empty_classification
            
        root_expr = ast_expr_list[-1].get("expr_tree", {})
        classification = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(root_expr)
        
        return classification

    @staticmethod
    def _parse_proof_state(goal_str: str, header: str = "") -> ProofState:
        """Parse Lean Infoview goal string into a ProofState (with inherited header)."""
        s = goal_str.strip()
        
        if not s or "Goals accomplished" in s or "no goals" in s:
            return ProofState(context="", goal="SOLVED_OR_EMPTY", header=header)

        parts = s.split("⊢")
        
        if len(parts) > 1:
            ctx_raw = parts[0].strip()
            main_goal_raw = parts[1].strip()
            
            goal_lines = []
            for line in main_goal_raw.splitlines():
                if line.startswith("case ") or "Goals accomplished" in line:
                    break
                goal_lines.append(line)
            goal = "\n".join(goal_lines).strip()
        else:
            ctx_raw, goal = "", s.strip()

        if goal.lower() == "sorry":
            goal = "ELABORATION_FAULT"

        valid_ctx_lines = []
        if ctx_raw:
            for line in ctx_raw.splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("case"):
                    valid_ctx_lines.append(line)
                    
        ctx = "\n".join(valid_ctx_lines)

        return ProofState(context=ctx, goal=goal, header=header)

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