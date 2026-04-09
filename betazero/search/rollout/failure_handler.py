from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.utils.lean_cmd import build_theorem
from betazero.utils.lean_parse import parse_proof_state
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .execution_result import LeanExecutionResult
from .utils import extract_action_body, inject_patched_code_to_raw


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
        action_code: str,
        result: LeanExecutionResult,
        prompt: str = "",
        *,
        is_sc_tactic: bool = False,
    ) -> None:
        """Timeout / crash / transport errors: penalize graph edge; do not run sorrifier."""
        sc = result.state_code
        if not sc:
            try:
                sc = build_theorem(state, action_code)
            except Exception:
                sc = action_code
        r = 0.0
        graph.expand(
            state,
            Action(
                action_type=action_kind,
                content=action_code,
                extracted_code="",
                children=(),
                prompt=prompt,
                is_sc_tactic=is_sc_tactic,
            ),
            r_env=r,
            tactic_status="FAILED" if action_kind == "tactic" else None,
        )

    def handle_failed_tactic(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
        *,
        is_sc_tactic: bool = False,
    ) -> str:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        graph.expand(
            state,
            Action(
                action_type="tactic",
                content=action_code,
                extracted_code=extract_action_body(patched),
                children=(),
                prompt=prompt,
                is_sc_tactic=is_sc_tactic,
            ),
            r_env=r_fail,
            tactic_status="FAILED",
        )
        return extract_action_body(patched)

    def handle_failed_skeleton(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
    ) -> None:
        patched_skeleton = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched_skeleton)
        patched_action_code = extract_action_body(patched_skeleton)
        r_fail = self.reward.r_env(state_code, patched_skeleton, patched_vr)
        graph.expand(
            state,
            Action(
                action_type="skeleton",
                content=action_code,
                extracted_code="",
                children=(),
                prompt=prompt,
            ),
            r_env=r_fail,
        )
        r_patch = self.reward.r_env(patched_skeleton, patched_skeleton, patched_vr)
        new_subgoals = [
            parse_proof_state(s.get("goal", ""), header=state.header)
            for s in patched_vr.get("sorries", [])
        ]
        graph.expand(
            state,
            Action(
                action_type="skeleton",
                content=inject_patched_code_to_raw(state_code, patched_skeleton),
                extracted_code=patched_action_code,
                children=tuple(new_subgoals),
                prompt=prompt,
            ),
            r_env=r_patch,
        )
