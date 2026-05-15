from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gammazero.core import ProofState, Action
from gammazero.utils.lean_cmd import build_theorem
from gammazero.utils.lean_parse import extract_proof_body, parse_proof_state
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import RewardCalculator

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
    patched: str
    patched_vr: dict
    patched_action_code: str
    r_fail: float
    new_subgoals: tuple[ProofState, ...]


class FailureHandler:
    """Sorrify failed tactics/skeletons and register penalized / patched graph edges."""

    def __init__(self, lean: LeanEnv, sorrifier: Sorrifier, reward: RewardCalculator):
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward

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
        graph.expand(
            state,
            Action(
                action_type=action_kind,
                content=action_content,
                extracted_code="",
                children=(),
                prompt=prompt,
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
        sorrifier = self._new_sorrifier()
        patched = sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        patched_action_code = extract_proof_body(patched)

        full_orig = build_theorem(state, lean_code)
        full_patched = build_theorem(state, patched_action_code)
        r_fail = self.reward.r_env(full_orig, full_patched, patched_vr)

        new_subgoals: tuple[ProofState, ...] = ()
        if action_kind == "skeleton":
            new_subgoals = tuple(
                parse_proof_state(s.get("goal", ""), header=state.header)
                for s in patched_vr.get("sorries", [])
            )

        return FailedActionPatch(
            state=state,
            action_kind=action_kind,
            action_content=action_content,
            lean_code=lean_code,
            prompt=prompt,
            patched=patched,
            patched_vr=patched_vr,
            patched_action_code=patched_action_code,
            r_fail=r_fail,
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
        )

        graph.expand(
            patch.state,
            failed_action,
            r_env=patch.r_fail,
            tactic_status="FAILED" if patch.action_kind == "tactic" else None,
        )

        return patch.patched_action_code
