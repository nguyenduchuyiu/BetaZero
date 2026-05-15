"""Assigns structural dependencies reward (r_dep) to skeleton actions."""

from gammazero.env.lean_env import LeanEnv
from gammazero.search.graph import ANDORGraph
from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.utils.lean_cmd import build_theorem
from .calculator import RewardCalculator
import re


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis to assign dependency rewards."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def _extract_sorry_vars(self, code: str) -> set[str]:
        """Parses Lean code indentation to accurately attribute 'sorry' to its declaring variable."""
        lines = code.splitlines()
        sorry_vars = set()
        stack = []
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            
            while stack and indent <= stack[-1][0]:
                stack.pop()
                
            match = re.match(r"(?:have|let)\s+([a-zA-Z0-9_]+)\s*[:=]", stripped)
            if match:
                var_name = match.group(1)
                stack.append((indent, var_name))
                
            if re.search(r"\bsorry\b", stripped):
                if stack:
                    sorry_vars.add(stack[-1][1])
                    
        return sorry_vars

    @staticmethod
    def _has_real_sorry(code: str) -> bool:
        """Return true if `sorry` appears outside Lean comments."""
        clean = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")
        return bool(re.search(r"\bsorry\b", clean))

    def _expand_verifiable_garbage(
        self,
        parent_state,
        stitched_code: str,
        candidates: set[str],
        garbage_vars: list[str],
    ) -> tuple[list[str], str | None]:
        """
        Greedily remove candidate skeleton variables when Lean confirms the
        stitched proof still compiles. This catches tactic-generated proof terms
        that mention an otherwise redundant local hypothesis.
        """
        ordered_garbage = list(dict.fromkeys(garbage_vars))
        unresolved_sorry_vars = self._extract_sorry_vars(stitched_code)
        cleaned_code = (
            ProofStitcher.prune_garbage(stitched_code, ordered_garbage)
            if ordered_garbage
            else stitched_code
        )

        trial_candidates = candidates - set(ordered_garbage) - unresolved_sorry_vars
        for var in sorted(trial_candidates):
            trial_code = ProofStitcher.prune_garbage(cleaned_code, [var])
            if self._has_real_sorry(trial_code):
                continue

            trial_full_code = build_theorem(parent_state, trial_code)
            trial_vr = self.lean.verify(trial_full_code)
            if trial_vr.get("complete"):
                ordered_garbage.append(var)
                cleaned_code = trial_code

        return ordered_garbage, cleaned_code

    def assign(self, graph: ANDORGraph) -> None:
        """Bottom-up dependency reward assignment using Expr Trees."""
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue

            if not action.children:
                continue
            
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            stitched_code = ProofStitcher.stitch(action.extracted_code, child_proofs)
            full_compilable_code = build_theorem(parent_state, stitched_code)
            
            allowed_vars = self._extract_sorry_vars(action.extracted_code)
            dep_analysis = self.lean.analyze_dependencies(full_compilable_code, allowed_vars=allowed_vars)
            
            r_dep_score = 0.0

            if action.extracted_code and graph.get_r_env(action) == 1.0:
                base_malignant = dep_analysis.get("malignant", [])
                base_benign = dep_analysis.get("benign", [])
                garbage_vars = base_malignant + base_benign
                garbage_vars, cleaned_test_code = self._expand_verifiable_garbage(
                    parent_state,
                    stitched_code,
                    allowed_vars,
                    garbage_vars,
                )
                
                if self._has_real_sorry(cleaned_test_code):
                    r_dep_score = 0.0
                else:
                    # Dependency analysis proposes garbage pruning; Lean remains
                    # the source of truth before a skeleton can be marked solved.
                    cleaned_full_code = build_theorem(parent_state, cleaned_test_code)
                    cleaned_vr = self.lean.verify(cleaned_full_code)
                    if cleaned_vr.get("complete"):
                        cleaned_dep_analysis = self.lean.analyze_dependencies(
                            cleaned_full_code,
                            allowed_vars=allowed_vars,
                        )
                        if len(cleaned_dep_analysis.get("core_failed", [])) > 0:
                            r_dep_score = 0.0
                        else:
                            base_garbage = set(base_malignant) | set(base_benign)
                            extra_benign = sorted(set(garbage_vars) - base_garbage)
                            r_dep_score = self.reward.r_dep({
                                "core": cleaned_dep_analysis.get("core_solved", []),
                                "benign": cleaned_dep_analysis.get("benign", []) + base_benign + extra_benign,
                                "malignant": cleaned_dep_analysis.get("malignant", []) + base_malignant,
                            })

                    if cleaned_vr.get("complete") and r_dep_score > 0:
                        if garbage_vars:
                            graph.set_garbage_vars(action, garbage_vars)
                        graph.set_skeleton_override(action, True)
                    else:
                        r_dep_score = 0.0

            graph.set_r_dep(action, r_dep_score)
