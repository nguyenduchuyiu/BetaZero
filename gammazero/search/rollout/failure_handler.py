from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gammazero.core import ProofState, Action
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.lean_parse import extract_proof_body, parse_proof_state
from gammazero.utils.scaffold import (
    isolate_sorry_target,
    replace_sorry_at,
    sorry_index_for_placeholder_index,
)
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import DependencyRewardAssigner, RewardCalculator

from .execution_result import LeanExecutionResult
if TYPE_CHECKING:
    from gammazero.env.lean_env import LeanEnv
    from gammazero.search.sorrifier import Sorrifier


@dataclass(frozen=True)
class FailedActionPatch:
    state: ProofState
    action_kind: str
    action_content: str
    lean_code: str
    prompt: str
    verify_code: str
    stitched_code: str
    patched: str
    patched_vr: dict
    patched_action_code: str
    lean_feedback: str
    r_fail: float
    r_dep: float
    new_subgoals: tuple[ProofState, ...]


class FailureHandler:
    """Sorrify failed tactics/skeletons and register penalized / patched graph edges."""

    def __init__(self, lean: LeanEnv, sorrifier: Sorrifier, reward: RewardCalculator):
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward
        self.reward_assigner = DependencyRewardAssigner(lean, reward)

    def handle_system_execute_failure(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_kind: str,
        action_content: str,
        result: LeanExecutionResult,
        prompt: str = "",
    ) -> None:
        """Timeout / crash / transport errors: penalize graph edge; do not run sorrifier."""
        r = 0.0
        print(
            f"[FailureHandler] System failure while executing {action_kind}: "
            f"{result.system_errors or 'unknown transport error'}",
            flush=True,
        )
        print(f"[FailureHandler] Goal: {state.goal[:240]}", flush=True)
        graph.expand(
            state,
            Action(
                action_type=action_kind,
                content=action_content,
                extracted_code="",
                children=(),
                prompt=prompt,
                verify_code=result.state_code,
                stitched_code="",
                patched_code="",
                lean_feedback=result.system_errors or "",
            ),
            r_env=r,
            tactic_status="FAILED" if action_kind == "tactic" else None,
        )

    def _new_sorrifier(self) -> Sorrifier:
        from gammazero.search.sorrifier import Sorrifier

        log_path = self.sorrifier.log_path
        if log_path:
            root, ext = os.path.splitext(log_path)
            log_path = f"{root}.{uuid.uuid4().hex}{ext or '.log'}"
        return Sorrifier(
            self.sorrifier.repl_verifier,
            max_cycles=self.sorrifier.max_cycles,
            log_path=log_path,
        )

    @staticmethod
    def _verify_code_for_state(state: ProofState, action_code: str) -> str:
        if state.scaffold_code:
            return replace_sorry_at(state.scaffold_code, state.target_index, action_code)
        return build_theorem(state, action_code)

    def compute_failed_action_patch(
        self,
        state: ProofState,
        action_kind: str,
        action_content: str,
        lean_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
    ) -> FailedActionPatch:
        """Run the expensive, side-effect-free patching work for a failed action."""
        try:
            sorrifier = self._new_sorrifier()
            patched = sorrifier.fix_code(state_code)
            patched_vr = self.lean.verify(patched)
            patched_action_code = extract_proof_body(patched)

            full_orig = self._verify_code_for_state(state, lean_code)
            full_patched = self._verify_code_for_state(state, patched_action_code)
            r_fail = self.reward.r_env(full_orig, full_patched, patched_vr)
            r_dep = 0.0
            if action_kind == "tactic" and patched_vr.get("pass"):
                r_dep = self.reward_assigner.calculate_patched_tactic_r_dep(
                    full_patched,
                    patched_action_code,
                )

            new_subgoals: tuple[ProofState, ...] = ()
            if action_kind == "skeleton":
                parsed_subgoals = []
                for sorry_idx, s in enumerate(patched_vr.get("sorries", [])):
                    target_index = sorry_index_for_placeholder_index(patched, sorry_idx)
                    if target_index is None:
                        continue
                    child_scaffold, child_target_index = isolate_sorry_target(
                        patched,
                        target_index,
                    )
                    ps = parse_proof_state(s.get("goal", ""), header=state.header)
                    parsed_subgoals.append(
                        ProofState(
                            context=ps.context,
                            goal=ps.goal,
                            header=ps.header,
                            scaffold_code=child_scaffold,
                            target_index=child_target_index,
                            target_kind="patched_skeleton_child",
                        )
                    )
                new_subgoals = tuple(parsed_subgoals)
            lean_feedback = ""
        except Exception as e:
            first_error = ""
            if state_vr.get("errors"):
                first_error = state_vr["errors"][0].get("data", "")
            print(
                f"[FailureHandler] Sorrifier failed while patching {action_kind}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            print(f"[FailureHandler] Goal: {state.goal[:240]}", flush=True)
            if first_error:
                print(f"[FailureHandler] Original Lean error: {first_error[:500]}", flush=True)
            patched = state_code
            patched_vr = {
                "pass": False,
                "complete": False,
                "errors": [],
                "warnings": [],
                "sorries": [],
                "system_errors": f"Sorrifier failed: {type(e).__name__}: {e}",
            }
            patched_action_code = "sorry"
            r_fail = 0.0
            r_dep = 0.0
            new_subgoals = ()
            lean_feedback = f"Sorrifier failed: {type(e).__name__}: {e}"

        return FailedActionPatch(
            state=state,
            action_kind=action_kind,
            action_content=action_content,
            lean_code=lean_code,
            prompt=prompt,
            verify_code=state_code,
            stitched_code=state_code,
            patched=patched,
            patched_vr=patched_vr,
            patched_action_code=patched_action_code,
            lean_feedback=lean_feedback,
            r_fail=r_fail,
            r_dep=r_dep,
            new_subgoals=new_subgoals,
        )

    def apply_failed_action_patch(self, graph: ANDORGraph, patch: FailedActionPatch) -> str:
        """Mutate the graph for a precomputed failed-action patch."""
        failed_action = Action(
            action_type=patch.action_kind,
            content=patch.action_content,
            extracted_code=patch.lean_code,
            children=(),
            prompt=patch.prompt,
            verify_code=patch.verify_code,
            stitched_code=patch.stitched_code,
            patched_code=patch.patched,
            lean_feedback=patch.lean_feedback,
        )

        graph.expand(
            patch.state,
            failed_action,
            r_env=patch.r_fail,
            r_dep=patch.r_dep,
            tactic_status="FAILED" if patch.action_kind == "tactic" else None,
        )

        return patch.patched_action_code
