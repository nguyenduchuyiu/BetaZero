from __future__ import annotations
from nodes import ProofState, Action


class ANDORGraph:
    """AND/OR search tree. OR-nodes = ProofState, AND-nodes = Action."""

    def __init__(self, root: ProofState):
        self._actions: dict[ProofState, list[Action]] = {root: []}
        self._parent:  dict[Action, ProofState] = {}
        self._r_env:   dict[Action, float] = {}
        self._r_dep:   dict[Action, float] = {}
        self._closed:  set[Action] = set()
        self._depth:   dict[ProofState, int] = {root: 0}

    def expand(self, state: ProofState, action: Action,
               r_env: float = 0.0, r_dep: float = 0.0, closed: bool = False):
        """Register an action under state and initialize its child states."""
        self._actions.setdefault(state, []).append(action)
        self._parent[action] = state
        self._r_env[action] = r_env
        self._r_dep[action] = r_dep
        if closed:
            self._closed.add(action)
        for child in action.children:
            self._actions.setdefault(child, [])
            self._depth.setdefault(child, self._depth[state] + 1)

    def get_actions(self, state: ProofState) -> list[Action]:
        return self._actions.get(state, [])

    def get_r_env(self, action: Action) -> float:
        return self._r_env.get(action, 0.0)

    def get_parent(self, action: Action, default: ProofState | None = None) -> ProofState | None:
        return self._parent.get(action, default)

    def parent_items(self):
        return self._parent.items()

    def set_r_dep(self, action: Action, r_dep: float):
        self._r_dep[action] = r_dep

    def get_depth(self, state: ProofState) -> int:
        return self._depth.get(state, -1)

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        """Recursive solved check with cycle detection."""
        if visiting is None:
            visiting = set()
        if node in visiting:
            return False
        visiting.add(node)
        try:
            if isinstance(node, ProofState):
                return any(self.is_solved(a, visiting) for a in self._actions.get(node, []))
            if node.action_type == "tactic":
                return node in self._closed
            # Skeleton: all children must be solved
            return bool(node.children) and all(self.is_solved(c, visiting) for c in node.children)
        finally:
            visiting.remove(node)

    def unsolved_states(self) -> list[ProofState]:
        return [s for s in self._actions if not self.is_solved(s)]

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        """Bottom-up Q-value computation per Section 6.3."""
        q_cache: dict[Action, float] = {}
        v_cache: dict[ProofState, float] = {}
        visiting: set = set()

        def V(state: ProofState) -> float:
            if state in v_cache:
                return v_cache[state]
            if state in visiting:
                return 0.0
            visiting.add(state)
            val = max((Q(a) for a in self._actions.get(state, [])), default=0.0)
            visiting.remove(state)
            v_cache[state] = val
            return val

        def Q(action: Action) -> float:
            if action in q_cache:
                return q_cache[action]
            r_env  = self._r_env.get(action, 0.0)
            solved = self.is_solved(action)
            if action.action_type == "tactic":
                val = r_env + W_solve * float(solved)
            else:
                r_dep  = self._r_dep.get(action, 0.0)
                future = gamma * min((V(c) for c in action.children), default=0.0)
                val = r_env + float(solved) * (r_dep + future)
            q_cache[action] = val
            return val

        for action in self._parent:
            Q(action)
        return q_cache


if __name__ == "__main__":
    root = ProofState("ctx0", "goal0")
    graph = ANDORGraph(root)

    state_a = ProofState("ctx_A", "goal_A")
    state_b = ProofState("ctx_B", "goal_B")
    action_skel = Action("skeleton", "have h_A have h_B", (state_a, state_b))
    graph.expand(root, action_skel, r_env=0.8)
    graph.set_r_dep(action_skel, 0.5)

    graph.expand(state_a, Action("tactic", "simp", ()), r_env=0.2, closed=False)
    graph.expand(state_a, Action("tactic", "omega", ()), r_env=1.0, closed=True)
    graph.expand(state_b, Action("tactic", "linarith", ()), r_env=1.0, closed=True)

    print(f"Root solved: {graph.is_solved(root)}")
    print(f"Depth of state_b: {graph.get_depth(state_b)}")
    print(f"Unsolved states: {graph.unsolved_states()}")
    q_vals = graph.backup(gamma=0.99, W_solve=1.0)
    print("\n--- Q-Values ---")
    for act, q in q_vals.items():
        print(f"Q({act.action_type:8s} | {act.content:18s}) = {q:.2f}")
