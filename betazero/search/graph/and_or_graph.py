from __future__ import annotations

import threading
from typing import Any, Literal

from betazero.core.nodes import Action, NodeStatus, ProofState


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

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        with self._lock:
            if visiting is None:
                visiting = set()
            if node in visiting:
                return False
            if node in self._solved_cache:
                return self._solved_cache[node]
            visiting.add(node)
            try:
                if isinstance(node, ProofState):
                    res = any(
                        self.is_solved(a, visiting)
                        for a in list(self._actions.get(node, []))
                    )
                elif node.action_type == "tactic":
                    res = self._tactic_status.get(node) == "SOLVED"
                else:
                    res = bool(node.children) and all(
                        self.is_solved(c, visiting) for c in node.children
                    )
                self._solved_cache[node] = res
                return res
            finally:
                visiting.remove(node)

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

    def get_depth(self, state: ProofState) -> int:
        with self._lock:
            return self._depth.get(state, -1)

    @staticmethod
    def _is_solved_from_snapshot(
        node: ProofState | Action,
        actions: dict[ProofState, list[Action]],
        tactic_status: dict[Action, Literal["SOLVED", "FAILED"]],
        visiting: set,
        memo: dict[Any, bool],
    ) -> bool:
        if node in visiting:
            return False
        if node in memo:
            return memo[node]
        visiting.add(node)
        try:
            if isinstance(node, ProofState):
                res = any(
                    ANDORGraph._is_solved_from_snapshot(a, actions, tactic_status, visiting, memo)
                    for a in actions.get(node, [])
                )
            elif node.action_type == "tactic":
                res = tactic_status.get(node) == "SOLVED"
            else:
                res = bool(node.children) and all(
                    ANDORGraph._is_solved_from_snapshot(c, actions, tactic_status, visiting, memo)
                    for c in node.children
                )
            memo[node] = res
            return res
        finally:
            visiting.remove(node)

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        with self._lock:
            actions = {k: list(v) for k, v in self._actions.items()}
            parent = dict(self._parent)
            r_env = dict(self._r_env)
            r_dep = dict(self._r_dep)
            tactic_status = dict(self._tactic_status)

        q_cache: dict[Action, float] = {}
        v_cache: dict[ProofState, float] = {}
        visiting_v: set[ProofState] = set()
        solve_memo: dict[Any, bool] = {}

        def is_solved_snap(n: ProofState | Action) -> bool:
            return self._is_solved_from_snapshot(n, actions, tactic_status, set(), solve_memo)

        def V(state: ProofState) -> float:
            if state in v_cache:
                return v_cache[state]
            if state in visiting_v:
                return 0.0
            visiting_v.add(state)
            val = max((Q(a) for a in actions.get(state, [])), default=0.0)
            visiting_v.remove(state)
            v_cache[state] = val
            return val

        def Q(action: Action) -> float:
            if action in q_cache:
                return q_cache[action]
            r_e = r_env.get(action, 0.0)
            solved = is_solved_snap(action)
            if action.action_type == "tactic":
                val = r_e + W_solve * float(solved)
            else:
                r_d = r_dep.get(action, 0.0)
                future = gamma * min((V(c) for c in action.children), default=0.0)
                val = r_e + float(solved) * (r_d + future)
            q_cache[action] = val
            return val

        for action in parent:
            Q(action)
        return q_cache
