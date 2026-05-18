from gammazero.core import ProofState
from gammazero.env import Lean4ServerScheduler
from gammazero.utils.lean_parse import parse_proof_state
from gammazero.utils.lean_cmd import build_theorem


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
        from gammazero.env.ast_parser import get_lean_ast

        return get_lean_ast(code)

    def analyze_dependencies(
        self,
        proof_code: str,
        allowed_vars: set[str] | None = None,
        target_name: str | None = None,
    ) -> dict:
        """
        Classify subgoals using Lean 4 Expr Tree deep analysis.
        Returns classifications for: core_solved, core_failed, malignant, benign.
        """
        from gammazero.env.expr_parser import get_lean_expr_tree
        from gammazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
        ast_expr_list = get_lean_expr_tree(proof_code)
        
        empty_classification = {
            "core_solved": [], "core_failed": [], "malignant": [], "benign": []
        }
        
        if not ast_expr_list:
            return empty_classification
            
        # Tìm block có theorem_name là "my_theorem" (mặc định của build_theorem)
        # Nếu không thấy, lấy block cuối cùng làm fallback.
        root_expr = {}
        for block in reversed(ast_expr_list):
            if block.get("theorem") == "my_theorem":
                root_expr = block.get("expr_value_tree", {})
                break
        
        if not root_expr and ast_expr_list:
            root_expr = ast_expr_list[-1].get("expr_value_tree", {})
            
        if not root_expr:
            return empty_classification
            
        classification = SHARED_EXPR_ANALYZER.classify_skeleton_subgoals(
            root_expr,
            allowed_vars=allowed_vars,
            target_name=target_name,
        )
        
        return classification
