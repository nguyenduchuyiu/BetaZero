from __future__ import annotations

from betazero.core.nodes import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.search.sorrifier import Sorrifier
from betazero.search.and_or_graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from typing import Protocol

class SamplePolicy(Protocol):
    def sample(self, states: list[ProofState], action_type: str, n: int) -> list[list[str]]: ...

TACTIC_RATIO = 0.8


class LevelwiseRollout:
    def __init__(self, policy: SamplePolicy, lean: LeanEnv,
                 sorrifier: Sorrifier, reward: RewardCalculator,
                 K: int = 32, max_depth: int = 5, max_nodes: int = 128):
        self.policy = policy
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward
        self.K = K
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def rollout(self, theorem: ProofState) -> list[tuple[ProofState, Action, float, float]]:
        """Run level-wise rollout; return (state, action, r_env, Q) per action."""
        graph = ANDORGraph(theorem)
        total_expanded = 0
        K_tac  = max(1, int(self.K * TACTIC_RATIO))
        K_skel = self.K - K_tac

        for depth in range(self.max_depth):
            frontier = [s for s in graph.unsolved_states() if graph.get_depth(s) == depth]
            if not frontier or total_expanded >= self.max_nodes:
                break
                
            # Sample half of budget for tactic actions and self-correction for remaining budget
            tac_batches  = self.policy.sample(frontier, "tactic", K_tac//2)
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
                    state_code, state_vr, subgoals = self.lean.execute(state, action_code)
                    if action_type == "tactic":
                        self._process_tactic(graph, state, action_code, state_code, state_vr)
                    else:
                        self._process_skeleton(graph, state, action_code, state_code, state_vr, subgoals)

        self._assign_dep_rewards(graph)
        q_values = self.reward.compute_returns(graph)
        return [(graph.get_parent(a, theorem), a, graph.get_r_env(a), q) for a, q in q_values.items()]

    def _extract_action_code(self, state_code: str) -> str:
        """Strip the `example ... := by` wrapper from a compiled state code string."""
        if ":= by\n" in state_code:
            body = state_code.split(":= by\n", 1)[1]
            return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())
        return state_code

    def _process_tactic(self, graph: ANDORGraph, state: ProofState,
                        action_code: str, state_code: str, state_vr: dict):
        if state_vr.get("complete"):
            r_env = self.reward.r_env(state_code, state_code, state_vr)
            graph.expand(state, Action("tactic", action_code, ()), r_env=r_env, closed=True)
        else:
            self._process_failed_tactic(graph, state, action_code, state_code)

    def _process_failed_tactic(self, graph: ANDORGraph, state: ProofState,
                               action_code: str, state_code: str) -> None:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        # Store original action code with penalty; tactic dead-end gets no children
        graph.expand(state, Action("tactic", action_code, ()), r_env=r_fail, closed=False)
        #TODO: self-correction for failed tactic

    def _process_skeleton(self, graph: ANDORGraph, state: ProofState, action_code: str,
                          state_code: str, state_vr: dict, subgoals: list[ProofState]):
        if state_vr.get("complete"):
            r_env = self.reward.r_env(state_code, state_code, state_vr)
            graph.expand(state, Action("skeleton", action_code, tuple(subgoals)), r_env=r_env)
        else:
            self._process_failed_skeleton(graph, state, action_code, state_code)

    def _process_failed_skeleton(self, graph: ANDORGraph, state: ProofState,
                                 action_code: str, state_code: str) -> None:
        patched = self.sorrifier.fix_code(state_code)
        patched_vr = self.lean.verify(patched)
        patched_action_code = self._extract_action_code(patched)
        r_fail = self.reward.r_env(state_code, patched, patched_vr)
        # Failed node: original code, penalized, no children
        graph.expand(state, Action("skeleton", action_code, ()), r_env=r_fail)
        # Patched node: sorrified code with valid subgoals, allowed to expand
        r_patch = self.reward.r_env(patched, patched, patched_vr)
        new_subgoals = [
            self.lean._parse_proof_state(s.get("goal", ""), header=state.header)
            for s in patched_vr.get("sorries", [])
        ]
        graph.expand(state, Action("skeleton", patched_action_code, tuple(new_subgoals)), r_env=r_patch)

    def _assign_dep_rewards(self, graph: ANDORGraph):
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue
            state_code = self.lean._build_cmd(parent_state, action.content)
            graph.set_r_dep(action, self.reward.r_dep(self.lean.dep_graph(state_code)))
