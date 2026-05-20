"""Score dependency rewards for skeleton and tactic actions."""

from gammazero.env.lean_env import LeanEnv
from gammazero.search.graph import ANDORGraph
from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.scaffold import replace_sorry_at
from .calculator import RewardCalculator
import re


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis for skeleton rewards."""

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

    @staticmethod
    def _full_code_for_state(state, proof_code: str) -> str:
        if getattr(state, "scaffold_code", ""):
            return replace_sorry_at(state.scaffold_code, state.target_index, proof_code)
        return build_theorem(state, proof_code)

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

            trial_full_code = self._full_code_for_state(parent_state, trial_code)
            trial_vr = self.lean.verify(trial_full_code)
            if trial_vr.get("complete"):
                ordered_garbage.append(var)
                cleaned_code = trial_code

        return ordered_garbage, cleaned_code

    def stitch_and_score_skeletons(self, graph: ANDORGraph) -> None:
        """Fill skeleton sorries with child proofs, then score dependency quality.

        Nested skeletons must be rescored after deeper skeletons become
        stitchable. A single parent-to-child pass can leave a parent with stale
        `sorry` placeholders even when all leaves are solved.
        """
        max_rounds = max(1, len(graph.all_actions()))
        for _ in range(max_rounds):
            changed = False
            for action, parent_state in self._skeleton_parent_items_bottom_up(graph):
                changed = self._stitch_and_score_one_skeleton(graph, action, parent_state) or changed
            if not changed:
                break

    def assign(self, graph: ANDORGraph) -> None:
        self.stitch_and_score_skeletons(graph)

    def _skeleton_parent_items_bottom_up(self, graph: ANDORGraph):
        return sorted(
            (
                (action, parent_state)
                for action, parent_state in graph.parent_items()
                if self._should_score_skeleton(action)
            ),
            key=lambda item: graph.get_depth(item[1]),
            reverse=True,
        )

    def _stitch_and_score_one_skeleton(self, graph: ANDORGraph, action, parent_state) -> bool:
        if not self._should_score_skeleton(action):
            return False

        stitched_code = self._stitch_child_proofs(graph, action)
        parent_skeleton_target = self._parent_skeleton_target(graph, parent_state)
        if parent_skeleton_target is not None:
            (
                grandparent_state,
                parent_skeleton,
                target_child_index,
            ) = parent_skeleton_target
            r_dep_score, garbage_vars, solved_by_stitched_proof = (
                self._score_stitched_subgoal_skeleton(
                    grandparent_state,
                    parent_skeleton,
                    target_child_index,
                    action.extracted_code,
                    stitched_code,
                    graph.get_r_env(action),
                )
            )
        else:
            r_dep_score, garbage_vars, solved_by_stitched_proof = self._score_stitched_skeleton(
                parent_state,
                action.extracted_code,
                stitched_code,
                graph.get_r_env(action),
            )

        changed = graph.set_stitched_code(action, stitched_code)
        if solved_by_stitched_proof:
            if garbage_vars:
                graph.set_garbage_vars(action, garbage_vars)
            changed = graph.set_skeleton_override(action, True) or changed
        changed = graph.set_r_dep(action, r_dep_score) or changed
        return changed

    @staticmethod
    def _should_score_skeleton(action) -> bool:
        return action.action_type == "skeleton" and bool(action.children)

    @staticmethod
    def _stitch_child_proofs(graph: ANDORGraph, action) -> str:
        child_proofs = [graph.extract_proof_code(child) for child in action.children]
        return ProofStitcher.stitch(action.extracted_code, child_proofs)

    @staticmethod
    def _parent_skeleton_target(
        graph: ANDORGraph,
        state,
    ):
        for parent_state, action, child_index in graph.parent_skeleton_items_for_state(state):
            if parent_state == state:
                continue
            if graph.status(action) not in ("FAILED", "RESERVED"):
                return parent_state, action, child_index
        return None

    @staticmethod
    def _skeleton_with_target_replacement(
        skeleton,
        target_child_index: int,
        replacement_code: str,
    ) -> str:
        child_proofs = [
            replacement_code if idx == target_child_index else "admit"
            for idx, _ in enumerate(skeleton.children)
        ]
        return ProofStitcher.stitch(skeleton.extracted_code, child_proofs)

    @staticmethod
    def _target_decl_name(skeleton_code: str, target_child_index: int) -> str | None:
        matches = list(re.finditer(r"\bsorry\b", skeleton_code or ""))
        if target_child_index < 0 or target_child_index >= len(matches):
            return None
        prefix = skeleton_code[: matches[target_child_index].start()]
        decl_matches = list(
            re.finditer(
                r"(?:^|\n)\s*(?:have|let)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
                prefix,
            )
        )
        if not decl_matches:
            return None
        return decl_matches[-1].group(1)

    def _score_stitched_skeleton(
        self,
        parent_state,
        skeleton_code: str,
        stitched_code: str,
        r_env: float,
    ) -> tuple[float, list[str], bool]:
        if not skeleton_code or r_env != 1.0:
            return 0.0, [], False

        skeleton_subgoal_vars = self._extract_sorry_vars(skeleton_code)
        initial_analysis = self._analyze_stitched_dependencies(
            parent_state,
            stitched_code,
            skeleton_subgoal_vars,
        )
        base_malignant = initial_analysis.get("malignant", [])
        base_benign = initial_analysis.get("benign", [])
        initial_garbage_vars = base_malignant + base_benign
        garbage_vars, cleaned_code = self._expand_verifiable_garbage(
            parent_state,
            stitched_code,
            skeleton_subgoal_vars,
            initial_garbage_vars,
        )

        if self._has_real_sorry(cleaned_code):
            return 0.0, garbage_vars, False

        r_dep_score, cleaned_complete = self._score_cleaned_stitched_proof(
            parent_state,
            cleaned_code,
            skeleton_subgoal_vars,
            base_benign,
            base_malignant,
            garbage_vars,
        )
        return (r_dep_score if cleaned_complete else 0.0), garbage_vars, cleaned_complete

    def _score_stitched_subgoal_skeleton(
        self,
        grandparent_state,
        parent_skeleton,
        target_child_index: int,
        mini_skeleton_code: str,
        stitched_mini_code: str,
        r_env: float,
    ) -> tuple[float, list[str], bool]:
        if not mini_skeleton_code or r_env != 1.0:
            return 0.0, [], False

        mini_subgoal_vars = self._extract_sorry_vars(mini_skeleton_code)
        parent_stitched_code = self._skeleton_with_target_replacement(
            parent_skeleton,
            target_child_index,
            stitched_mini_code,
        )
        if self._has_real_sorry(stitched_mini_code):
            return 0.0, [], False

        target_name = self._target_decl_name(parent_skeleton.extracted_code, target_child_index)
        if target_name is None:
            return 0.0, [], False

        full_code = self._full_code_for_state(grandparent_state, parent_stitched_code)
        verified = self.lean.verify(full_code)
        if not verified.get("pass"):
            return 0.0, [], False

        dep_analysis = self.lean.analyze_dependencies(
            full_code,
            allowed_vars=mini_subgoal_vars,
            target_name=target_name,
        )
        r_dep_score = self._score_dependency_analysis(dep_analysis)
        return r_dep_score, [], True

    def _analyze_stitched_dependencies(
        self,
        parent_state,
        stitched_code: str,
        target_vars: set[str],
    ) -> dict:
        full_code = self._full_code_for_state(parent_state, stitched_code)
        return self.lean.analyze_dependencies(full_code, allowed_vars=target_vars)

    def _score_cleaned_stitched_proof(
        self,
        parent_state,
        cleaned_code: str,
        target_vars: set[str],
        base_benign: list[str],
        base_malignant: list[str],
        garbage_vars: list[str],
    ) -> tuple[float, bool]:
        # Dependency analysis proposes garbage pruning; Lean remains the source
        # of truth before a skeleton can be marked solved.
        cleaned_full_code = self._full_code_for_state(parent_state, cleaned_code)
        cleaned_vr = self.lean.verify(cleaned_full_code)
        if not cleaned_vr.get("complete"):
            return 0.0, False

        cleaned_dep_analysis = self.lean.analyze_dependencies(
            cleaned_full_code,
            allowed_vars=target_vars,
        )
        if cleaned_dep_analysis.get("core_failed"):
            return 0.0, True

        base_garbage = set(base_malignant) | set(base_benign)
        extra_benign = sorted(set(garbage_vars) - base_garbage)
        return self.reward.r_dep({
            "core": cleaned_dep_analysis.get("core_solved", []),
            "benign": cleaned_dep_analysis.get("benign", []) + base_benign + extra_benign,
            "malignant": cleaned_dep_analysis.get("malignant", []) + base_malignant,
        }), True

    def calculate_r_dep(
        self,
        full_code: str,
        action_code: str,
        target_name: str | None = None,
    ) -> float:
        """Score whether a tactic consumes the local `have/let`s it introduced."""
        action_local_vars = self._extract_action_local_vars(action_code)

        if not action_local_vars and not self._has_real_sorry(full_code):
            return 1.0

        dep_analysis = self.lean.analyze_dependencies(
            full_code,
            allowed_vars=action_local_vars,
            target_name=target_name,
        )
        if target_name is not None:
            return self._score_dependency_analysis(dep_analysis)

        garbage_vars = dep_analysis.get("benign", []) + dep_analysis.get("malignant", [])
        cleaned_action_code = self._prune_verifiable_tactic_garbage(
            full_code,
            action_code,
            action_local_vars,
            garbage_vars,
        )
        if cleaned_action_code == action_code:
            return self._score_dependency_analysis(dep_analysis)

        cleaned_full_code = self._replace_full_code_body(full_code, cleaned_action_code)
        cleaned_dep_analysis = self.lean.analyze_dependencies(
            cleaned_full_code,
            allowed_vars=action_local_vars,
            target_name=target_name,
        )
        return self._score_dependency_analysis(cleaned_dep_analysis)

    def calculate_patched_tactic_r_dep(
        self,
        full_code: str,
        action_code: str,
        target_name: str | None = None,
    ) -> float:
        """
        Score a patched tactic that may contain local `have ... := sorry`
        scaffolding. Naked/exact sorry remains fatal, but a sorry-backed local
        fact can receive credit when the final proof depends on it.
        """
        action_local_vars = self._extract_action_local_vars(action_code)
        if self._has_unsafe_tactic_sorry(action_code):
            return 0.0
        if not action_local_vars and not self._has_real_sorry(full_code):
            return 1.0

        dep_analysis = self.lean.analyze_dependencies(
            full_code,
            allowed_vars=action_local_vars,
            target_name=target_name,
        )
        return self._score_patched_dependency_analysis(dep_analysis)

    def _prune_verifiable_tactic_garbage(
        self,
        full_code: str,
        action_code: str,
        action_local_vars: set[str],
        garbage_vars: list[str],
    ) -> str:
        cleaned_code = ProofStitcher.prune_garbage(action_code, list(dict.fromkeys(garbage_vars)))
        if not self._verify_full_code_body(full_code, cleaned_code):
            cleaned_code = action_code

        pruned_vars = set(garbage_vars) if cleaned_code != action_code else set()
        for var in sorted(action_local_vars - pruned_vars):
            trial_code = ProofStitcher.prune_garbage(cleaned_code, [var])
            if trial_code == cleaned_code or self._has_real_sorry(trial_code):
                continue
            if self._verify_full_code_body(full_code, trial_code):
                cleaned_code = trial_code
                pruned_vars.add(var)

        return cleaned_code

    def _verify_full_code_body(self, full_code: str, action_code: str) -> bool:
        try:
            return bool(self.lean.verify(self._replace_full_code_body(full_code, action_code)).get("complete"))
        except Exception:
            return False

    @staticmethod
    def _replace_full_code_body(full_code: str, action_code: str) -> str:
        prefix, sep, _ = full_code.partition(":= by")
        if not sep:
            return full_code

        lines = action_code.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        indented = "\n".join(f"  {line}" for line in lines)
        return f"{prefix}{sep}\n{indented}\n"

    def _extract_action_local_vars(self, code: str) -> set[str]:
        """Extract named local `have`/`let` binders introduced by an action."""
        clean = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")
        local_vars = set()
        for line in clean.splitlines():
            match = re.match(r"\s*(?:have|let)\s+([a-zA-Z_][a-zA-Z0-9_']*)\b", line)
            if match:
                local_vars.add(match.group(1))
        return local_vars

    def _has_unsafe_tactic_sorry(self, code: str) -> bool:
        """Return true when `sorry` is not inside a local have/let proof."""
        clean = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", code or "")
        stack = []
        for line in clean.splitlines():
            stripped = line.lstrip()
            if not stripped:
                continue

            indent = len(line) - len(stripped)
            while stack and indent <= stack[-1][0]:
                stack.pop()

            match = re.match(r"(?:have|let)\s+([a-zA-Z_][a-zA-Z0-9_']*)\b", stripped)
            if match:
                stack.append((indent, match.group(1)))

            if re.search(r"\bsorry\b", stripped) and not stack:
                return True
        return False

    def _score_dependency_analysis(self, dep_analysis: dict) -> float:
        if dep_analysis.get("core_failed"):
            return 0.0

        return self.reward.r_dep({
            "core": dep_analysis.get("core_solved", []),
            "benign": dep_analysis.get("benign", []),
            "malignant": dep_analysis.get("malignant", []),
        })

    def _score_patched_dependency_analysis(self, dep_analysis: dict) -> float:
        failed_core = [x for x in dep_analysis.get("core_failed", []) if x != "MAIN_GOAL"]
        if "MAIN_GOAL" in dep_analysis.get("core_failed", []):
            return 0.0

        return self.reward.r_dep({
            "core": dep_analysis.get("core_solved", []) + failed_core,
            "benign": dep_analysis.get("benign", []),
            "malignant": dep_analysis.get("malignant", []),
        })
