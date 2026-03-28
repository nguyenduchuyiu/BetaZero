from __future__ import annotations
from nodes import ProofState, Action

class ANDORGraph:
    """
    AND/OR search tree for theorem proving.
    OR-nodes = ProofState
    AND-nodes = Action
    """
    def __init__(self, root: ProofState):
        self._actions: dict[ProofState, list[Action]] = {root: []}
        self._parent: dict[Action, ProofState] = {}
        self._r_env: dict[Action, float] = {}
        self._r_dep: dict[Action, float] = {}
        self._closed: set[Action] = set()
        # O(1) depth lookup caching
        self._depth: dict[ProofState, int] = {root: 0}

    def expand(self, state: ProofState, action: Action,
               r_env: float = 0.0, r_dep: float = 0.0, closed: bool = False):
        """Link an action to a state and initialize child states."""
        self._actions.setdefault(state, []).append(action)
        self._parent[action] = state
        self._r_env[action] = r_env
        self._r_dep[action] = r_dep
        
        if closed:
            self._closed.add(action)
            
        for child in action.children:
            self._actions.setdefault(child, [])
            # Inherit parent depth + 1
            self._depth.setdefault(child, self._depth[state] + 1)

    def set_r_dep(self, action: Action, r_dep: float):
        """Update dependency reward after trajectory rollout."""
        self._r_dep[action] = r_dep

    def get_depth(self, state: ProofState) -> int:
        """O(1) depth lookup."""
        return self._depth.get(state, -1)

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        """Check if node is solved, using cycle detection to prevent recursion errors."""
        if visiting is None:
            visiting = set()
            
        if node in visiting:
            return False  # Break cycle
            
        visiting.add(node)
        
        try:
            if isinstance(node, ProofState):
                # OR-node: True if any child action is solved
                return any(self.is_solved(a, visiting) for a in self._actions.get(node, []))
            
            if node.action_type == "tactic":
                # Terminal tactic is solved if explicitly marked closed
                return node in self._closed
                
            # Skeleton AND-node: True only if all child subgoals are solved
            return bool(node.children) and all(self.is_solved(c, visiting) for c in node.children)
        finally:
            visiting.remove(node)

    def unsolved_states(self) -> list[ProofState]:
        """Return all generated states that remain unsolved."""
        return [s for s in self._actions if not self.is_solved(s)]

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        """Recursive bottom-up Q-value computation with cycle breaking."""
        q_cache: dict[Action, float] = {}
        v_cache: dict[ProofState, float] = {}
        visiting: set = set()

        def V(state: ProofState) -> float:
            if state in v_cache:
                return v_cache[state]
            if state in visiting:
                return 0.0  # Break cycle with neutral value
                
            visiting.add(state)
            actions = self._actions.get(state, [])
            # V(s) = max_a Q(s, a)
            val = max((Q(a) for a in actions), default=0.0)
            visiting.remove(state)
            
            v_cache[state] = val
            return val

        def Q(action: Action) -> float:
            if action in q_cache:
                return q_cache[action]
                
            r_env = self._r_env.get(action, 0.0)
            solved = self.is_solved(action)
            
            if action.action_type == "tactic":
                val = r_env + W_solve * float(solved)
            else:
                r_dep = self._r_dep.get(action, 0.0)
                # Skeleton value strictly bounded by worst-case subgoal
                future = gamma * min((V(c) for c in action.children), default=0.0)
                val = r_env + float(solved) * (r_dep + future)
                
            q_cache[action] = val
            return val

        for action in self._parent:
            Q(action)

        return q_cache

if __name__ == "__main__":
    # --- Test Setup ---
    root = ProofState("ctx0", "goal0")
    graph = ANDORGraph(root)

    # Action 1 (Skeleton): Decomposes root into state A and state B
    state_a = ProofState("ctx_A", "goal_A")
    state_b = ProofState("ctx_B", "goal_B")
    action_skel = Action("skeleton", "have h_A have h_B", (state_a, state_b))
    
    graph.expand(root, action_skel, r_env=0.8)
    graph.set_r_dep(action_skel, 0.5)

    # Action 2 (Tactic): Fails to solve state A
    action_fail = Action("tactic", "simp", ())
    graph.expand(state_a, action_fail, r_env=0.2, closed=False)

    # Action 3 (Tactic): Successfully solves state A
    action_solve_a = Action("tactic", "omega", ())
    graph.expand(state_a, action_solve_a, r_env=1.0, closed=True)

    # Action 4 (Tactic): Successfully solves state B
    action_solve_b = Action("tactic", "linarith", ())
    graph.expand(state_b, action_solve_b, r_env=1.0, closed=True)

    # --- Verification ---
    print(f"Root solved: {graph.is_solved(root)}")
    print(f"Depth of state_b: {graph.get_depth(state_b)}")
    print(f"Unsolved states: {graph.unsolved_states()}")

    # Backup Q-values
    q_vals = graph.backup(gamma=0.99, W_solve=1.0)
    print("\n--- Q-Values ---")
    for act, q in q_vals.items():
        print(f"Q({act.action_type:8s} | {act.content:18s}) = {q:.2f}")