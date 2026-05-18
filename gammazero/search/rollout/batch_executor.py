from __future__ import annotations

import concurrent.futures
import re
import textwrap
import threading
from typing import TYPE_CHECKING

from gammazero.core import ProofState, Action
from gammazero.policy.output_parser import (
    get_lean_code,
    get_subgoal_skeleton_code,
    get_subgoal_tactic_code,
    strip_lean_comments,
)
from gammazero.policy.prompt import build_prompt
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import DependencyRewardAssigner, RewardCalculator
from gammazero.search.sorrifier.stitcher import ProofStitcher
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.lean_parse import parse_proof_state

from .execution_result import LeanExecutionResult
from .failure_handler import FailedActionPatch, FailureHandler
from .utils import format_lean_feedback

if TYPE_CHECKING:
    from gammazero.env.lean_env import LeanEnv


class RolloutBudget:
    __slots__ = ("max_nodes", "used", "_lock")

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.used = 0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.used = 0

    def try_consume(self) -> bool:
        with self._lock:
            if self.used >= self.max_nodes:
                return False
            self.used += 1
            return True


class BatchExecutor:
    """Parallel Lean execute + expand graph; tactic feedbacks align with action_batches[i][j]."""

    BAD_FINAL_GOAL_SKELETON_FEEDBACK = (
        "SKELETON POLICY VIOLATION: this skeleton introduces a `sorry` leaf whose goal "
        "is the original parent goal. This matches the BAD EXAMPLE pattern "
        "`have h_final : <original final goal> := sorry; exact h_final`. "
        "Do not restate the original goal as a leaf obligation. Decompose it into "
        "strictly smaller intermediate obligations, and make the final assembly "
        "sorry-free."
    )
    FORBIDDEN_TACTIC_FEEDBACK = (
        "TACTIC POLICY VIOLATION: tactic proof bodies must not contain `sorry` "
        "or `admit` outside comments. The subgoal verifier may contain sibling "
        "`admit` placeholders, so the replacement code itself must be placeholder-free."
    )

    def __init__(
        self,
        lean: LeanEnv,
        failure_handler: FailureHandler,
        reward: RewardCalculator,
        max_workers: int | None = None,
    ):
        self.lean = lean
        self.failure = failure_handler
        self.reward = reward
        self.reward_assigner = DependencyRewardAssigner(lean, reward)
        # Get max workers from the executor to synchronize, avoid context switching.
        ex = getattr(lean.scheduler, "executor", None)
        self._max_workers = max_workers if max_workers is not None else (
            getattr(ex, "_max_workers", 4) if ex is not None else 4
        )

    @staticmethod
    def safe_execute(lean: LeanEnv, state: ProofState, action_code: str) -> LeanExecutionResult:
        """Run Lean; never raises — transport/executor errors become `system_errors` on the result."""
        try:
            sc, vr, sg = lean.execute(state, action_code)
            return LeanExecutionResult.ok(sc, vr, sg)
        except Exception as e:
            try:
                sc = build_theorem(state, action_code)
            except Exception:
                sc = ""
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}", sc)

    @staticmethod
    def _has_forbidden_tactic_token(action_code: str) -> bool:
        clean = strip_lean_comments(action_code)
        return bool(re.search(r"\b(?:sorry|admit)\b", clean))

    @staticmethod
    def _subgoal_tactic_target(
        graph: ANDORGraph,
        state: ProofState,
    ) -> tuple[ProofState, Action, int] | None:
        for parent_state, action, child_index in graph.parent_skeleton_items_for_state(state):
            if graph.status(action) != "FAILED":
                return parent_state, action, child_index
        return None

    _subgoal_skeleton_target = _subgoal_tactic_target

    @staticmethod
    def _subgoal_skeleton_with_replacement(
        skeleton: Action,
        target_child_index: int,
        action_code: str,
    ) -> str:
        child_proofs: list[str | None] = []
        for idx, _ in enumerate(skeleton.children):
            child_proofs.append(action_code if idx == target_child_index else "admit")
        return ProofStitcher.stitch(skeleton.extracted_code, child_proofs)

    @staticmethod
    def _subgoal_target_decl_name(skeleton: Action, target_child_index: int) -> str | None:
        matches = list(re.finditer(r"\bsorry\b", skeleton.extracted_code or ""))
        if target_child_index < 0 or target_child_index >= len(matches):
            return None
        prefix = skeleton.extracted_code[: matches[target_child_index].start()]
        decl_matches = list(
            re.finditer(
                r"(?:^|\n)\s*(?:have|let)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
                prefix,
            )
        )
        if not decl_matches:
            return None
        return decl_matches[-1].group(1)

    @staticmethod
    def _strip_optional_by(action_code: str) -> str:
        action_code = textwrap.dedent(action_code).strip("\n")
        match = re.match(r"^\s*by(?:\s+|$)", action_code)
        if not match:
            return action_code
        return textwrap.dedent(action_code[match.end():]).strip("\n")

    @staticmethod
    def _extract_between_markers(code: str, start_marker: str, end_marker: str) -> str:
        lines = code.splitlines()
        start = next(i for i, line in enumerate(lines) if start_marker in line)
        end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if end_marker in line)
        return textwrap.dedent("\n".join(lines[start + 1 : end])).strip("\n")

    @classmethod
    def _subgoal_target_score_code(
        cls,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        action_code: str,
    ) -> str:
        start_marker = "-- GAMMAZERO_TARGET_SCORE_START"
        end_marker = "-- GAMMAZERO_TARGET_SCORE_END"
        target_body = cls._strip_optional_by(action_code)
        marked_target = f"{start_marker}\n{target_body}\n{end_marker}"
        marked_body = cls._subgoal_skeleton_with_replacement(
            skeleton,
            target_child_index,
            marked_target,
        )
        marked_full = build_theorem(parent_state, marked_body)
        target_lines = cls._extract_between_markers(marked_full, start_marker, end_marker)
        return build_theorem(parent_state, target_lines)

    @staticmethod
    def safe_execute_subgoal_tactic(
        lean: LeanEnv,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        action_code: str,
    ) -> LeanExecutionResult:
        try:
            skeleton_code = BatchExecutor._subgoal_skeleton_with_replacement(
                skeleton,
                target_child_index,
                action_code,
            )
            candidate_code = build_theorem(parent_state, skeleton_code)
            vr = lean.verify(candidate_code)
            return LeanExecutionResult.ok(candidate_code, vr, [])
        except Exception as e:
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}")

    @staticmethod
    def safe_execute_subgoal_skeleton(
        lean: LeanEnv,
        child_state: ProofState,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        action_code: str,
    ) -> LeanExecutionResult:
        try:
            skeleton_code = BatchExecutor._subgoal_skeleton_with_replacement(
                skeleton,
                target_child_index,
                action_code,
            )
            candidate_code = build_theorem(parent_state, skeleton_code)
            vr = lean.verify(candidate_code)
            excluded_goals = {
                BatchExecutor._normalize_goal(child.goal)
                for i, child in enumerate(skeleton.children)
                if i != target_child_index
            }
            subgoals: list[ProofState] = []
            for s in vr.get("sorries", []):
                ps = parse_proof_state(s.get("goal", ""), header=child_state.header)
                if ps.goal in ["SOLVED_OR_EMPTY", "ELABORATION_FAULT"]:
                    continue
                if BatchExecutor._normalize_goal(ps.goal) in excluded_goals:
                    continue
                subgoals.append(ps)
            return LeanExecutionResult.ok(candidate_code, vr, subgoals)
        except Exception as e:
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}")

    def compute_failed_subgoal_tactic_patch(
        self,
        child_state: ProofState,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        raw_output: str,
        action_code: str,
        candidate_code: str,
        prompt: str,
    ) -> FailedActionPatch:
        """Patch a failed subgoal tactic in parent context, then score target lines only."""
        sorrifier = self.failure._new_sorrifier()
        patched = sorrifier.fix_code(candidate_code)
        patched_vr = self.lean.verify(patched)
        patched_raw = f"```lean4\n{patched}\n```"
        patched_action_code = get_subgoal_tactic_code(
            patched_raw,
            skeleton.extracted_code,
            target_child_index,
        )
        if not patched_action_code:
            patched_action_code = "sorry"

        full_orig = self._subgoal_target_score_code(
            parent_state,
            skeleton,
            target_child_index,
            action_code,
        )
        full_patched = self._subgoal_target_score_code(
            parent_state,
            skeleton,
            target_child_index,
            patched_action_code,
        )
        r_fail = self.reward.r_env(full_orig, full_patched, patched_vr)
        r_dep = 0.0
        if patched_vr.get("pass"):
            target_name = self._subgoal_target_decl_name(skeleton, target_child_index)
            if target_name is not None:
                r_dep = self.reward_assigner.calculate_patched_tactic_r_dep(
                    patched,
                    patched_action_code,
                    target_name=target_name,
                )

        return FailedActionPatch(
            state=child_state,
            action_kind="tactic",
            action_content=raw_output,
            lean_code=action_code,
            prompt=prompt,
            patched=patched,
            patched_vr=patched_vr,
            patched_action_code=patched_action_code,
            r_fail=r_fail,
            r_dep=r_dep,
            new_subgoals=(),
        )

    def compute_failed_subgoal_skeleton_patch(
        self,
        child_state: ProofState,
        parent_state: ProofState,
        skeleton: Action,
        target_child_index: int,
        raw_output: str,
        action_code: str,
        candidate_code: str,
        prompt: str,
    ) -> FailedActionPatch:
        """Patch a failed mini-skeleton in parent context, then score target lines only."""
        sorrifier = self.failure._new_sorrifier()
        patched = sorrifier.fix_code(candidate_code)
        patched_vr = self.lean.verify(patched)
        patched_raw = f"```lean4\n{patched}\n```"
        patched_action_code = get_subgoal_skeleton_code(
            patched_raw,
            skeleton.extracted_code,
            target_child_index,
        )
        if not patched_action_code:
            patched_action_code = "sorry"

        full_orig = self._subgoal_target_score_code(
            parent_state,
            skeleton,
            target_child_index,
            action_code,
        )
        full_patched = self._subgoal_target_score_code(
            parent_state,
            skeleton,
            target_child_index,
            patched_action_code,
        )
        r_fail = self.reward.r_env(full_orig, full_patched, patched_vr)

        return FailedActionPatch(
            state=child_state,
            action_kind="skeleton",
            action_content=raw_output,
            lean_code=action_code,
            prompt=prompt,
            patched=patched,
            patched_vr=patched_vr,
            patched_action_code=patched_action_code,
            r_fail=r_fail,
            r_dep=0.0,
            new_subgoals=(),
        )

    @staticmethod
    def _normalize_goal(goal: str) -> str:
        return " ".join(goal.split())

    @classmethod
    def skeleton_restates_parent_goal(
        cls,
        state: ProofState,
        subgoals: list[ProofState],
    ) -> bool:
        parent_goal = cls._normalize_goal(state.goal)
        return any(cls._normalize_goal(subgoal.goal) == parent_goal for subgoal in subgoals)

    def execute(
        self,
        graph: ANDORGraph,
        states: list[ProofState],
        action_batches: list[list[dict]],
        action_type: str,
        budget: RolloutBudget,
        prompts: list[str] | None = None,
    ) -> list[list[tuple[str, str, str] | None]]:
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]

        tasks: list[
            tuple[
                int,
                int,
                ProofState,
                str,
                str,
                bool,
                bool,
                concurrent.futures.Future,
            ]
        ] = []
        feedbacks: list[list[tuple[str, str, str] | None]] = [
            [None] * len(actions) for actions in action_batches
        ]
        patch_futures: list[
            tuple[int, int, str, dict, concurrent.futures.Future[FailedActionPatch]]
        ] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for i, (state, actions) in enumerate(zip(states, action_batches)):
                for j, action_dict in enumerate(actions):
                    if not budget.try_consume():
                        break
                    raw_output = action_dict["text"]
                    subgoal_tactic = False
                    subgoal_skeleton = False
                    target = None
                    if action_type == "tactic":
                        target = self._subgoal_tactic_target(graph, state)
                    elif action_type == "skeleton":
                        target = self._subgoal_skeleton_target(graph, state)
                    if target is not None:
                        _, skeleton, target_child_index = target
                        if action_type == "tactic":
                            subgoal_tactic = True
                            lean_code = get_subgoal_tactic_code(
                                raw_output,
                                skeleton.extracted_code,
                                target_child_index,
                            )
                        else:
                            subgoal_skeleton = True
                            lean_code = get_subgoal_skeleton_code(
                                raw_output,
                                skeleton.extracted_code,
                                target_child_index,
                            )
                    else:
                        lean_code = get_lean_code(raw_output, allow_body=action_type == "tactic")
                    if not lean_code:
                        self.failure.handle_system_execute_failure(
                            graph,
                            state,
                            action_type,
                            raw_output,
                            LeanExecutionResult.from_transport_error("empty_lean_code"),
                            prompts[i],
                        )
                        continue
                    if action_type == "tactic" and self._has_forbidden_tactic_token(lean_code):
                        graph.expand(
                            state,
                            Action(
                                action_type="tactic",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=(),
                                prompt=prompts[i],
                            ),
                            r_env=0.0,
                            tactic_status="FAILED",
                        )
                        feedbacks[i][j] = (lean_code, self.FORBIDDEN_TACTIC_FEEDBACK, "")
                        continue

                    if target is not None and action_type == "tactic":
                        parent_state, skeleton, target_child_index = target
                        fut = pool.submit(
                            BatchExecutor.safe_execute_subgoal_tactic,
                            self.lean,
                            parent_state,
                            skeleton,
                            target_child_index,
                            lean_code,
                        )
                    elif target is not None and action_type == "skeleton":
                        parent_state, skeleton, target_child_index = target
                        fut = pool.submit(
                            BatchExecutor.safe_execute_subgoal_skeleton,
                            self.lean,
                            state,
                            parent_state,
                            skeleton,
                            target_child_index,
                            lean_code,
                        )
                    else:
                        fut = pool.submit(BatchExecutor.safe_execute, self.lean, state, lean_code)
                    tasks.append((i, j, state, raw_output, lean_code, subgoal_tactic, subgoal_skeleton, fut))
                if budget.used >= budget.max_nodes:
                    break

            for i, j, state, raw_output, lean_code, subgoal_tactic, subgoal_skeleton, future in tasks:
                res: LeanExecutionResult = future.result()
                prompt = prompts[i]
                if res.has_system_failure:
                    self.failure.handle_system_execute_failure(
                        graph, state, action_type, raw_output, res, prompt
                    )
                    continue
                state_code, state_vr, subgoals = res.state_code, res.verify, list(res.subgoals)
                full_code = build_theorem(state, lean_code)
                # For a complete tactic, code passed Lean with 0 sorries → r_env = 1.0
                r_env = 1.0

                if action_type == "tactic" and subgoal_tactic:
                    if state_vr.get("pass"):
                        target = self._subgoal_tactic_target(graph, state)
                        r_dep = 0.0
                        if target is not None:
                            parent_state, skeleton, target_child_index = target
                            target_name = self._subgoal_target_decl_name(
                                skeleton,
                                target_child_index,
                            )
                            if target_name is not None:
                                r_dep = self.reward_assigner.calculate_r_dep(
                                    state_code,
                                    lean_code,
                                    target_name=target_name,
                                )
                        act = Action(
                            action_type="tactic",
                            content=raw_output,
                            extracted_code=lean_code,
                            children=(),
                            prompt=prompt,
                        )
                        graph.expand(
                            state,
                            act,
                            r_env=r_env,
                            r_dep=r_dep,
                            tactic_status="SOLVED",
                        )
                    else:
                        target = self._subgoal_tactic_target(graph, state)
                        if target is not None:
                            parent_state, skeleton, target_child_index = target
                            fut = pool.submit(
                                self.compute_failed_subgoal_tactic_patch,
                                state,
                                parent_state,
                                skeleton,
                                target_child_index,
                                raw_output,
                                lean_code,
                                state_code,
                                prompt,
                            )
                            patch_futures.append((i, j, lean_code, state_vr, fut))
                        else:
                            graph.expand(
                                state,
                                Action(
                                    action_type="tactic",
                                    content=raw_output,
                                    extracted_code=lean_code,
                                    children=(),
                                    prompt=prompt,
                                ),
                                r_env=0.0,
                                r_dep=0.0,
                                tactic_status="FAILED",
                            )
                            feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), "")
                    continue

                if action_type == "skeleton" and subgoal_skeleton:
                    target = self._subgoal_skeleton_target(graph, state)
                    if target is None:
                        graph.expand(
                            state,
                            Action(
                                action_type="skeleton",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=(),
                                prompt=prompt,
                            ),
                            r_env=0.0,
                        )
                        feedbacks[i][j] = (lean_code, "missing parent skeleton target", "")
                        continue

                    parent_state, skeleton, target_child_index = target
                    if state_vr.get("pass"):
                        if self.skeleton_restates_parent_goal(state, subgoals):
                            graph.expand(
                                state,
                                Action(
                                    action_type="skeleton",
                                    content=raw_output,
                                    extracted_code=lean_code,
                                    children=(),
                                    prompt=prompt,
                                ),
                                r_env=0.0,
                            )
                            feedbacks[i][j] = (
                                lean_code,
                                self.BAD_FINAL_GOAL_SKELETON_FEEDBACK,
                                lean_code,
                            )
                            continue

                        full_target = self._subgoal_target_score_code(
                            parent_state,
                            skeleton,
                            target_child_index,
                            lean_code,
                        )
                        r_env_score = self.reward.r_env(full_target, full_target, state_vr)
                        graph.expand(
                            state,
                            Action(
                                action_type="skeleton",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=tuple(subgoals),
                                prompt=prompt,
                            ),
                            r_env=r_env_score,
                        )
                    else:
                        fut = pool.submit(
                            self.compute_failed_subgoal_skeleton_patch,
                            state,
                            parent_state,
                            skeleton,
                            target_child_index,
                            raw_output,
                            lean_code,
                            state_code,
                            prompt,
                        )
                        patch_futures.append((i, j, lean_code, state_vr, fut))
                        feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), "")
                    continue

                if state_vr.get("complete"):
                    if action_type not in ("tactic", "skeleton"):
                        raise ValueError(f"Invalid action type: {action_type}")
                    r_dep = (
                        self.reward_assigner.calculate_r_dep(state_code, lean_code)
                        if action_type == "tactic"
                        else 0.0
                    )
                    act = Action(
                        action_type=action_type,
                        content=raw_output,
                        extracted_code=lean_code,
                        children=(),
                        prompt=prompt,
                    )
                    graph.expand(
                        state,
                        act,
                        r_env=r_env,
                        r_dep=r_dep,
                        tactic_status="SOLVED" if action_type == "tactic" else None,
                    )
                    if action_type == "skeleton":
                        graph.set_skeleton_override(act, True)
                elif action_type == "tactic":
                    fut = pool.submit(
                        self.failure.compute_failed_action_patch,
                        state,
                        action_type,
                        raw_output,
                        lean_code,
                        state_code,
                        state_vr,
                        prompt,
                    )
                    patch_futures.append((i, j, lean_code, state_vr, fut))
                    
                elif action_type == "skeleton":
                    if state_vr.get("pass"):
                        if self.skeleton_restates_parent_goal(state, subgoals):
                            graph.expand(
                                state,
                                Action(
                                    action_type="skeleton",
                                    content=raw_output,
                                    extracted_code=lean_code,
                                    children=(),
                                    prompt=prompt,
                                ),
                                r_env=0.0,
                            )
                            feedbacks[i][j] = (
                                lean_code,
                                self.BAD_FINAL_GOAL_SKELETON_FEEDBACK,
                                lean_code,
                            )
                            continue
                        # Calculate r_env even for passing skeletons to catch semantic/AST issues
                        r_env_score = self.reward.r_env(full_code, full_code, state_vr)
                        graph.expand(
                            state,
                            Action(
                                action_type="skeleton",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=tuple(subgoals),
                                prompt=prompt,
                            ),
                            r_env=r_env_score,
                        )
                    else:
                        fut = pool.submit(
                            self.failure.compute_failed_action_patch,
                            state,
                            action_type,
                            raw_output,
                            lean_code,
                            state_code,
                            state_vr,
                            prompt,
                        )
                        patch_futures.append((i, j, lean_code, state_vr, fut))
                else:
                    raise ValueError(f"Invalid action type: {action_type}")

            for i, j, lean_code, state_vr, future in patch_futures:
                patch = future.result()
                sorr_body = self.failure.apply_failed_action_patch(graph, patch)
                feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), sorr_body)

        return feedbacks
