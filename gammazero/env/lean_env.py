from gammazero.core import ProofState
from gammazero.env import Lean4ServerScheduler
from gammazero.utils.lean_parse import parse_proof_state
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.scaffold import (
    isolate_sorry_target,
    replace_sorry_at,
    verifier_sorries_by_source_position,
)


class LeanEnv:
    """Interface between proof search and the Lean verifier."""

    def __init__(self, scheduler: Lean4ServerScheduler):
        self.scheduler = scheduler

    def verify(self, code: str) -> dict:
        return self.scheduler.verify(code)

    def execute(self, state: ProofState, code: str) -> tuple[str, dict, list[ProofState]]:
        """Build, verify, and parse subgoals for a tactic applied to state."""
        start_marker = "-- GAMMAZERO_START"
        end_marker = "-- GAMMAZERO_END"
        marked_code = f"{start_marker}\n{code}\n{end_marker}"
        if state.scaffold_code:
            marked_candidate_code = replace_sorry_at(state.scaffold_code, state.target_index, marked_code)
        else:
            marked_candidate_code = build_theorem(state, marked_code)

        before_marker, after_start = marked_candidate_code.split(start_marker, 1)
        marked_target, after_marker = after_start.split(end_marker, 1)
        before_marker = before_marker.rstrip(" \t")
        marked_target = marked_target.strip("\n")
        candidate_code = before_marker + marked_target + after_marker

        vr = self.scheduler.verify(candidate_code)

        subgoals = []
        if vr.get("pass"):
            start_char = len(before_marker)
            end_char = start_char + len(marked_target)
            for target_index, s in verifier_sorries_by_source_position(
                candidate_code,
                vr.get("sorries", []),
                start_offset=start_char,
                end_offset=end_char,
            ):
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
        """Classify subgoals via Lean 4 expression-tree analysis.

        Returns lists for each class: core_solved, core_failed, malignant, benign.
        """
        from gammazero.env.expr_parser import get_lean_expr_tree
        from gammazero.search.sorrifier.dependency_analyzer import SHARED_EXPR_ANALYZER
        ast_expr_list = get_lean_expr_tree(proof_code)
        
        empty_classification = {
            "core_solved": [], "core_failed": [], "malignant": [], "benign": []
        }
        
        if not ast_expr_list:
            return empty_classification
            
        # Locate the block named "my_theorem" (the default from build_theorem).
        # Fall back to the last block if the name is not found.
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
