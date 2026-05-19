from gammazero.core import ProofState
from gammazero.env import Lean4ServerScheduler
from gammazero.utils.lean_parse import parse_proof_state
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.scaffold import (
    isolate_sorry_target,
    placeholder_count,
    replace_sorry_at,
    sorry_index_for_placeholder_index,
    verifier_placeholder_index_for_sorry,
)


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        if state.scaffold_code:
            candidate_code = replace_sorry_at(state.scaffold_code, state.target_index, code)
        else:
            candidate_code = build_theorem(state, code)
        vr = self.scheduler.verify(candidate_code)
        
        subgoals = []
        if vr.get("pass"):
            if state.scaffold_code:
                target_sorry_start = verifier_placeholder_index_for_sorry(
                    state.scaffold_code,
                    state.target_index,
                )
                target_sorry_end = target_sorry_start + placeholder_count(code)
            else:
                target_sorry_start = 0
                target_sorry_end = len(vr.get("sorries", []))
            for idx, s in enumerate(vr.get("sorries", [])):
                if idx < target_sorry_start or idx >= target_sorry_end:
                    continue
                target_index = sorry_index_for_placeholder_index(candidate_code, idx)
                if target_index is None:
                    continue
                child_scaffold, child_target_index = isolate_sorry_target(
                    candidate_code,
                    target_index,
                )
                ps = parse_proof_state(s.get("goal", ""), header=state.header)
                if ps.goal not in ["SOLVED_OR_EMPTY", "ELABORATION_FAULT"]:
                    subgoals.append(
                        ProofState(
                            context=ps.context,
                            goal=ps.goal,
                            header=ps.header,
                            scaffold_code=child_scaffold,
                            target_index=child_target_index,
                            target_kind="skeleton_child",
                        )
                    )

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
