from __future__ import annotations

import threading
from typing import Any, Literal

from betazero.core.nodes import Action, NodeStatus, ProofState
from betazero.policy.output_parser import get_lean_code
from betazero.search.sorrifier.stitcher import ProofStitcher


class ANDORGraph:
    """Thread-safe AND/OR proof graph with solved-state checks and return backup."""

    def __init__(self, root: ProofState):
        self._lock = threading.RLock()
        self._actions: dict[ProofState, list[Action]] = {root: []}
        self._parent: dict[Action, ProofState] = {}
        self._r_env: dict[Action, float] = {}
        self._r_dep: dict[Action, float] = {}
        self._tactic_status: dict[Action, Literal["SOLVED", "FAILED"]] = {}
        self._depth: dict[ProofState, int] = {root: 0}
        self._solved_cache: dict[Any, bool] = {}
        self._skeleton_override: dict[Action, bool] = {} 

    def expand(
        self,
        state: ProofState,
        action: Action,
        r_env: float = 0.0,
        r_dep: float = 0.0,
        tactic_status: Literal["SOLVED", "FAILED"] | None = None,
    ) -> None:
        with self._lock:
            if action in self._parent:
                return
            self._solved_cache.clear()
            self._actions.setdefault(state, []).append(action)
            self._parent[action] = state
            self._r_env[action] = r_env
            self._r_dep[action] = r_dep
            if tactic_status is not None and action.action_type == "tactic":
                self._tactic_status[action] = tactic_status
            for child in action.children:
                self._actions.setdefault(child, [])
                if child not in self._depth:
                    self._depth[child] = self._depth[state] + 1

    def _node_solved(
        self, node: ProofState | Action, visiting: set, memo: dict[Any, bool]
    ) -> bool:
        if node in visiting:
            return False
        if node in memo:
            return memo[node]
        visiting.add(node)
        try:
            if isinstance(node, ProofState):
                res = any(self._node_solved(a, visiting, memo) for a in self._actions.get(node, []))
            elif node.action_type == "tactic":
                res = self._tactic_status.get(node) == "SOLVED"
            else:
                if node in self._skeleton_override:
                    res = self._skeleton_override[node]
                else:
                    res = bool(node.children) and all(
                        self._node_solved(c, visiting, memo) for c in node.children
                    )
            memo[node] = res
            return res
        finally:
            visiting.remove(node)

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        with self._lock:
            if visiting is None:
                visiting = set()
            return self._node_solved(node, visiting, self._solved_cache)

    def status(self, node: ProofState | Action) -> NodeStatus:
        with self._lock:
            if isinstance(node, ProofState):
                return "SOLVED" if self.is_solved(node) else "OPEN"
            if node.action_type == "tactic":
                t = self._tactic_status.get(node)
                if t == "SOLVED":
                    return "SOLVED"
                if t == "FAILED":
                    return "FAILED"
                return "OPEN"
            if self.is_solved(node):
                return "SOLVED"
            if not node.children:
                return "FAILED"
            return "OPEN"

    def unsolved_states(self) -> list[ProofState]:
        with self._lock:
            keys = list(self._actions.keys())
        return [s for s in keys if not self.is_solved(s)]

    def get_actions(self, state: ProofState) -> list[Action]:
        with self._lock:
            return list(self._actions.get(state, []))

    def get_r_env(self, action: Action) -> float:
        with self._lock:
            return self._r_env.get(action, 0.0)

    def get_parent(self, action: Action, default: ProofState | None = None) -> ProofState | None:
        with self._lock:
            return self._parent.get(action, default)

    def parent_items(self) -> list[tuple[Action, ProofState]]:
        with self._lock:
            return list(self._parent.items())

    def set_r_dep(self, action: Action, r_dep: float) -> None:
        with self._lock:
            self._r_dep[action] = r_dep

    def set_skeleton_override(self, action: Action, is_solved: bool):
        with self._lock:
            self._skeleton_override[action] = is_solved
            self._solved_cache.clear() # Nhớ xóa cache để graph tính lại từ đầu

    def get_depth(self, state: ProofState) -> int:
        with self._lock:
            return self._depth.get(state, -1)

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        with self._lock:
            q_cache: dict[Action, float] = {}
            v_cache: dict[ProofState, float] = {}
            visiting_v: set[ProofState] = set()
            solve_memo: dict[Any, bool] = {}

            def V(state: ProofState) -> float:
                if state in v_cache:
                    return v_cache[state]
                if state in visiting_v:
                    return 0.0
                visiting_v.add(state)
                val = max((Q(a) for a in self._actions.get(state, [])), default=0.0)
                visiting_v.remove(state)
                v_cache[state] = val
                return val

            def Q(action: Action) -> float:
                if action in q_cache:
                    return q_cache[action]
                r_e = self._r_env.get(action, 0.0)
                solved = self._node_solved(action, set(), solve_memo)
                if action.action_type == "tactic":
                    val = r_e + W_solve * float(solved)
                else:
                    r_d = self._r_dep.get(action, 0.0)
                    future = gamma * min((V(c) for c in action.children), default=0.0)
                    val = r_e + float(solved) * (r_d + future)
                q_cache[action] = val
                return val

            for action in self._parent:
                Q(action)
            for state in self._actions:
                V(state)
            return q_cache

    def get_successful_action(self, state: ProofState) -> Action | None:
        """Retrieve the action that successfully solved this state."""
        with self._lock:
            for action in self.get_actions(state):
                if self.status(action) == "SOLVED":
                    return action
        return None

    def extract_proof_code(self, state: ProofState) -> str | None:
        """Recursively extract and stitch the successful proof code for a state."""
        
        action = self.get_successful_action(state)
        if not action:
            return None
            
        parsed_code = get_lean_code(action.content)
        
        if action.action_type == "tactic":
            return parsed_code
            
        # Skeleton: recurse down to children
        child_proofs = [self.extract_proof_code(child) for child in action.children]
        return ProofStitcher.stitch(parsed_code, child_proofs)