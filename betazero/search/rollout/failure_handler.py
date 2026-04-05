from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .execution_result import LeanExecutionResult
from .utils import extract_action_body


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
    ) -> None:
        """Timeout / crash / transport errors: penalize graph edge; do not run sorrifier."""
        sc = result.state_code
        if not sc:
            try:
                sc = self.lean._build_cmd(state, action_code)
            except Exception:
                sc = action_code
        vr = result.verify
        r = self.reward.r_env(sc, sc, vr)
        graph.expand(
            state,
            Action(action_kind, action_code, (), prompt=prompt),
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
    ) -> str:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        graph.expand(
            state, Action("tactic", action_code, (), prompt=prompt), r_env=r_fail, tactic_status="FAILED"
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
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        patched_action_code = extract_action_body(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        graph.expand(state, Action("skeleton", action_code, (), prompt=prompt), r_env=r_fail)
        r_patch = self.reward.r_env(patched, patched, patched_vr)
        new_subgoals = [
            self.lean._parse_proof_state(s.get("goal", ""), header=state.header)
            for s in patched_vr.get("sorries", [])
        ]
        graph.expand(
            state,
            Action("skeleton", patched_action_code, tuple(new_subgoals), prompt=prompt),
            r_env=r_patch,
        )
