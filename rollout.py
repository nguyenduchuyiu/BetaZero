from __future__ import annotations
from typing import Protocol

from nodes import ProofState, Action
from lean_env import LeanEnv
from lean_verifier import verify_lean_code
from sorrifier import Sorrifier
from and_or_graph import ANDORGraph
from reward import RewardCalculator

TACTIC_RATIO = 0.8

class PolicyModel(Protocol):
    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]:
        ...

class LevelwiseRollout:
    def __init__(self, policy: PolicyModel, lean: LeanEnv,
                 sorrifier: Sorrifier, reward: RewardCalculator,
                 K: int = 32, max_depth: int = 5, max_nodes: int = 128,
                 verify_timeout: int = 60):
        self.policy = policy
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward
        self.K = K
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.verify_timeout = verify_timeout

    def rollout(self, theorem: ProofState) -> list[tuple[ProofState, Action, float]]:
        graph = ANDORGraph(theorem)
        total_expanded = 0
        K_tac = max(1, int(self.K * TACTIC_RATIO))
        K_skel = self.K - K_tac

        for depth in range(self.max_depth):
            frontier = [s for s in graph.unsolved_states() if graph.get_depth(s) == depth]
            if not frontier or total_expanded >= self.max_nodes:
                break

            tac_batches = self.policy.sample(frontier, "tactic", K_tac)
            skel_batches = self.policy.sample(frontier, "skeleton", K_skel)

            for i, state in enumerate(frontier):
                if total_expanded >= self.max_nodes:
                    break

                candidates = [("tactic", c) for c in tac_batches[i]] + \
                             [("skeleton", c) for c in skel_batches[i]]

                for action_type, action_code in candidates:
                    if total_expanded >= self.max_nodes:
                        break
                    total_expanded += 1

                    # 1. Bọc khung lấy state_code và chạy thử
                    state_code, state_vr, subgoals = self.lean.execute(state, action_code)

                    # 2. Phân luồng
                    if action_type == "tactic":
                        self._process_tactic(graph, state, action_code, state_code, state_vr)
                    else:
                        self._process_skeleton(graph, state, action_code, state_code, state_vr, subgoals)
                            
        self._assign_dep_rewards(graph)
        q_values = self.reward.compute_returns(graph)

        return [
            (graph._parent.get(a, theorem), a, q)
            for a, q in q_values.items()
        ]

    # ------------------------------------------------------------------
    # Node Processing Helpers
    # ------------------------------------------------------------------

    def _extract_action_code(self, state_code: str) -> str:
        """Bóc phần ruột action ra khỏi lớp vỏ state_code (example...:= by)"""
        if ":= by\n" in state_code:
            body = state_code.split(":= by\n", 1)[1]
            return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
        return state_code

    def _process_tactic(self, graph: ANDORGraph, state: ProofState, action_code: str, 
                        state_code: str, state_vr: dict):
        if state_vr.get("complete"):  
            r_env = self.reward.r_env(state_code, state_code, state_vr)
            action = Action("tactic", action_code, ())
            graph.expand(state, action, r_env=r_env, closed=True)
        else:
            self._process_failed_tactic(graph, state, action_code, state_code)

    def _process_failed_tactic(self, graph: ANDORGraph, state: ProofState, 
                               action_code: str, state_code: str) -> None:
        patched_state_code = self.sorrifier.fix_code(state_code)
        patched_vr = verify_lean_code(patched_state_code, timeout=self.verify_timeout)
        r_fail = self.reward.r_env(state_code, patched_state_code, patched_vr)
        
        # Tactic xịt: Lưu action_code gốc để phạt điểm, không cho đi tiếp
        action_fail = Action("tactic", action_code, ())
        graph.expand(state, action_fail, r_env=r_fail, closed=False)

    def _process_skeleton(self, graph: ANDORGraph, state: ProofState, action_code: str, 
                          state_code: str, state_vr: dict, subgoals: list[ProofState]):
        if state_vr.get("complete"): 
            r_env = self.reward.r_env(state_code, state_code, state_vr)
            action = Action("skeleton", action_code, tuple(subgoals))
            graph.expand(state, action, r_env=r_env, closed=False)
        else:
            self._process_failed_skeleton(graph, state, action_code, state_code)

    def _process_failed_skeleton(self, graph: ANDORGraph, state: ProofState, 
                                 action_code: str, state_code: str) -> None:
        patched_state_code = self.sorrifier.fix_code(state_code)
        patched_vr = verify_lean_code(patched_state_code, timeout=self.verify_timeout)
        
        # BÓC TÁCH: Lấy ruột để nhét vào node vá
        patched_action_code = self._extract_action_code(patched_state_code)
        
        # Node 1: Thằng con hư (Lưu action_code gốc, Phạt)
        r_fail = self.reward.r_env(state_code, patched_state_code, patched_vr)
        action_fail = Action("skeleton", action_code, ())
        graph.expand(state, action_fail, r_env=r_fail, closed=False)

        # Node 2: Đứa con nuôi (Lưu patched_action_code, Cho đi tiếp)
        r_patch = self.reward.r_env(patched_state_code, patched_state_code, patched_vr)
        new_subgoals = [self.lean._parse_proof_state(s.get("goal", "")) 
                        for s in patched_vr.get("sorries", [])]
        action_patch = Action("skeleton", patched_action_code, tuple(new_subgoals))
        graph.expand(state, action_patch, r_env=r_patch, closed=False)
        
    def _assign_dep_rewards(self, graph: ANDORGraph):
        for action, parent_state in graph._parent.items():
            if action.action_type != "skeleton":
                continue
            state_code = self.lean._build_cmd(parent_state, action.content)
            dep = self.lean.dep_graph(state_code)
            graph.set_r_dep(action, self.reward.r_dep(dep))