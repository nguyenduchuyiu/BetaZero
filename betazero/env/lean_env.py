from betazero.core import ProofState
from betazero.env.ast_parser import get_lean_ast
from betazero.env import Lean4ServerScheduler
from betazero.env.expr_parser import get_lean_expr_tree
from betazero.utils.lean_parse import parse_proof_state
from betazero.utils.lean_cmd import build_theorem


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        candidate_code = build_theorem(state, code)
        vr = self.scheduler.verify(candidate_code)
        
        subgoals = []
        if vr.get("pass"):
            for s in vr.get("sorries", []):
                ps = parse_proof_state(s.get("goal", ""), header=state.header)
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
