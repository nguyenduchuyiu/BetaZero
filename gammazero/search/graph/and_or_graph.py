from __future__ import annotations

import threading
from typing import Any, Literal
import re

from gammazero.core.nodes import Action, NodeStatus, ProofState
from gammazero.policy.output_parser import get_lean_code
from gammazero.search.sorrifier.stitcher import ProofStitcher



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
        self._status_override: dict[ProofState | Action, NodeStatus] = {}
        self._garbage_vars: dict[Action, list[str]] = {}

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
            override = self._status_override.get(node)
            if override == "SOLVED":
                memo[node] = True
                return True
            if override == "FAILED":
                memo[node] = False
                return False
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
            override = self._status_override.get(node)
            if override is not None:
                return override
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

    def add_state(self, state: ProofState, depth: int | None = None) -> None:
        with self._lock:
            self._actions.setdefault(state, [])
            if depth is not None and state not in self._depth:
                self._depth[state] = depth

    def all_states(self) -> list[ProofState]:
        with self._lock:
            return list(self._actions.keys())

    def all_actions(self) -> list[Action]:
        with self._lock:
            return list(self._parent.keys())

    def mark_open(self, node: ProofState | Action) -> bool:
        with self._lock:
            old = self._status_override.pop(node, None)
            self._solved_cache.clear()
            return old is not None

    def mark_solved(self, node: ProofState | Action) -> bool:
        with self._lock:
            old = self._status_override.get(node)
            self._status_override[node] = "SOLVED"
            self._solved_cache.clear()
            return old != "SOLVED"

    def mark_failed(self, node: ProofState | Action) -> bool:
        with self._lock:
            old = self._status_override.get(node)
            self._status_override[node] = "FAILED"
            if isinstance(node, Action) and node.action_type == "tactic":
                self._tactic_status[node] = "FAILED"
            self._solved_cache.clear()
            return old != "FAILED"

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
                    live_children = self._live_children_for_action(action)
                    future = gamma * min((V(c) for c in live_children), default=0.0)
                    val = r_e + float(solved) * (r_d + future)
                q_cache[action] = val
                return val

            for action in self._parent:
                Q(action)
            for state in self._actions:
                V(state)
            return q_cache

    def get_successful_actions(self, state: ProofState) -> list[Action]:
        """Retrieve all actions that successfully solved this state."""
        with self._lock:
            return [a for a in self.get_actions(state) if self.status(a) == "SOLVED"]

    def get_successful_action(self, state: ProofState) -> Action | None:
        """Retrieve the first action that successfully solved this state."""
        actions = self.get_successful_actions(state)
        return actions[0] if actions else None

    def set_garbage_vars(self, action: Action, garbage_vars: list[str]):
        with self._lock:
            self._garbage_vars[action] = garbage_vars

    def get_garbage_vars(self, action: Action) -> list[str]:
        with self._lock:
            return self._garbage_vars.get(action, [])

    @staticmethod
    def _extract_sorry_var_order(code: str) -> list[str]:
        vars_in_order = []
        seen = set()
        stack = []

        for line in code.splitlines():
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)

            while stack and indent <= stack[-1][0]:
                stack.pop()

            match = re.match(r"(?:have|let)\s+([a-zA-Z0-9_]+)\s*[:=]", stripped)
            if match:
                stack.append((indent, match.group(1)))

            if re.search(r"\bsorry\b", stripped) and stack:
                var_name = stack[-1][1]
                if var_name not in seen:
                    vars_in_order.append(var_name)
                    seen.add(var_name)

        return vars_in_order

    def _live_children_for_action(self, action: Action) -> tuple[ProofState, ...]:
        garbage_vars = set(self._garbage_vars.get(action, []))
        if not garbage_vars:
            return action.children

        child_vars = self._extract_sorry_var_order(action.extracted_code)
        if len(child_vars) != len(action.children):
            return action.children

        return tuple(
            child for var_name, child in zip(child_vars, action.children)
            if var_name not in garbage_vars
        )

    def _extract_for_action(self, action: Action, visiting: set[ProofState]) -> str | None:
        """Extract proof code for a single action (tactic or skeleton)."""
        if action.action_type == "tactic":
            return action.extracted_code

        # Skeleton: recurse down to children
        child_proofs = [self._extract_proof_code(child, visiting) for child in action.children]
        
        stitched = ProofStitcher.stitch(action.extracted_code, child_proofs)
        
        # Dùng máy xén tỉa dựa trên list rác lưu trong graph
        garbage_vars = self.get_garbage_vars(action)
        if garbage_vars:
            stitched = ProofStitcher.prune_garbage(stitched, garbage_vars)
            
        return stitched

    def _extract_proof_code(self, state: ProofState, visiting: set[ProofState]) -> str | None:
        """Internal recursive extraction with cycle detection."""
        if state in visiting:
            return None  # Cycle detected — treat as unsolved
        visiting.add(state)
        try:
            solved_actions = self.get_successful_actions(state)
            if not solved_actions:
                return None

            fallback: str | None = None
            for action in solved_actions:
                proof = self._extract_for_action(action, visiting)
                if proof is None:
                    continue
                
                # Check if clean: strip comments before searching for 'sorry'
                clean_check = re.sub(r"/\-(?:.|\n)*?\-/|--.*", "", proof)
                if not re.search(r'\bsorry\b', clean_check):
                    return proof  # Clean proof found
                if fallback is None:
                    fallback = proof  # Keep first sorry-containing as fallback

            return fallback
        finally:
            visiting.discard(state)

    def extract_proof_code(self, state: ProofState) -> str | None:
        """Recursively extract and stitch the successful proof code for a state.
        
        Tries all SOLVED actions and prefers the first one producing a sorry-free proof.
        Falls back to the first sorry-containing proof if no clean proof exists.
        """
        return self._extract_proof_code(state, set())
