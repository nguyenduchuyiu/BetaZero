from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.utils.lean_cmd import build_theorem

from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .execution_result import LeanExecutionResult
from .utils import extract_action_body, inject_patched_code_to_raw
from betazero.utils.lean_parse import parse_proof_state

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

    def handle_failed_action(
        self,
        graph: ANDORGraph,
        state: ProofState,
        action_kind: str,
        action_content: str,
        lean_code: str,
        state_code: str,
        state_vr: dict,
        prompt: str = "",
    ) -> str:
        """Vá lỗi: Node gốc bị phong ấn (FAILED), sinh thêm Node phụ (OPEN) để đi tiếp nếu là Skeleton."""
        # 1. Gọi thợ vá lỗi Sorrifier
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        patched_action_code = extract_action_body(patched)
        
        full_orig = build_theorem(state, lean_code)
        full_patched = build_theorem(state, patched_action_code)
        r_fail = self.reward.r_env(full_orig, full_patched, patched_vr)

        # ---------------------------------------------------------
        # NHÁNH 1: BẢN GỐC (Dành cho RL GRPO học từ sai lầm)
        # ---------------------------------------------------------
        failed_action = Action(
            action_type=action_kind,
            content=action_content,
            extracted_code=lean_code,
            children=(),  
            prompt=prompt,
        )

        graph.expand(
            state,
            failed_action,
            r_env=r_fail,
            tactic_status="FAILED" if action_kind == "tactic" else None, 
        )

        # 2. Tạo bản vá (Mọc nhánh mới)
        if action_kind == "skeleton":
            new_subgoals = [
                parse_proof_state(s.get("goal", ""), header=state.header)
                for s in patched_vr.get("sorries", [])
            ]
            
            if new_subgoals:
                patched_raw_content = inject_patched_code_to_raw(action_content, patched)
                
                synthetic_action = Action(
                    action_type="skeleton",
                    content=patched_raw_content,
                    extracted_code=patched_action_code,
                    children=tuple(new_subgoals),
                    # ---> DÁN ID CỦA BẢN GỐC VÀO ĐÂY <---
                    prompt=f"[SYNTHETIC_PATCH] from {failed_action.id}", 
                )
                # obviously r_env=1.0, just for visualization in the graph, not the reward
                graph.expand(state, synthetic_action, r_env=1.0) 
                
        return patched_action_code