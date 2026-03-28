"""
LevelwiseRollout — BFS-level AND/OR proof search orchestrator.

Budget split per node: 80% tactic / 20% skeleton (hardcoded).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from nodes import ProofState, Action
from and_or_graph import ANDORGraph
from lean_env import LeanEnv
from lean_verifier import verify_lean_code, Lean4ServerScheduler
from reward import RewardCalculator
from sorrifier import Sorrifier

TACTIC_RATIO = 0.8


class PolicyModel(ABC):
    """Abstract policy: maps prompts → generated code strings."""

    @abstractmethod
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Return one completion per prompt (same length as input)."""
        ...


class LevelwiseRollout:
    """
    For each depth level d:
      1. Collect all unsolved OR-nodes at depth d.
      2. Batch-generate tactic/skeleton candidates via PolicyModel.
      3. Execute via LeanEnv._build_cmd + verify_lean_code.
      4. If errors → Sorrifier.fix_code() → re-verify.
      5. ANDORGraph.expand() with r_env / r_dep / closed.
    After max_depth, backup Q-values and return (state, action, Q) triples.
    """

    def __init__(
        self,
        root: ProofState,
        policy: PolicyModel,
        lean_env: LeanEnv,
        sorrifier_verifier: Lean4ServerScheduler,
        reward_calc: RewardCalculator,
        total_budget: int = 10,
        max_depth: int = 5,
        sorrifier_cycles: int = 30,
        sorrifier_timeout: int = 60,
    ):
        self.graph = ANDORGraph(root)
        self.policy = policy
        self.lean_env = lean_env
        self.sorrifier_verifier = sorrifier_verifier
        self.reward_calc = reward_calc
        self.max_depth = max_depth
        self.sorrifier_cycles = sorrifier_cycles
        self.sorrifier_timeout = sorrifier_timeout
        # 80 / 20 hardcoded
        self.n_tactic = round(total_budget * TACTIC_RATIO)
        self.n_skeleton = total_budget - self.n_tactic

    def run(self) -> list[tuple[ProofState, Action, float]]:
        """Execute BFS rollout; return (state, action, Q) for training."""
        for depth in range(self.max_depth):
            pending = [
                s for s in self.graph.unsolved_states()
                if self.graph.get_depth(s) == depth
            ]
            if not pending:
                break
            self._process_depth(pending)

        q_vals = self.reward_calc.compute_returns(self.graph)
        return [(self.graph._parent[a], a, q) for a, q in q_vals.items()]

    # ------------------------------------------------------------------

    def _process_depth(self, states: list[ProofState]):
        """Single batch call to policy covering all states at this depth."""
        tactic_prompts = [
            self._make_prompt(s, "tactic")
            for s in states
            for _ in range(self.n_tactic)
        ]
        skeleton_prompts = [
            self._make_prompt(s, "skeleton")
            for s in states
            for _ in range(self.n_skeleton)
        ]

        codes = self.policy.generate_batch(tactic_prompts + skeleton_prompts)
        t_codes = codes[:len(tactic_prompts)]
        s_codes = codes[len(tactic_prompts):]

        for i, state in enumerate(states):
            for code in t_codes[i * self.n_tactic:(i + 1) * self.n_tactic]:
                self._try_expand(state, code, "tactic")
            for code in s_codes[i * self.n_skeleton:(i + 1) * self.n_skeleton]:
                self._try_expand(state, code, "skeleton")

    def _try_expand(self, state: ProofState, code: str, action_type: str):
        """Execute code on state, patch if broken, then expand the graph."""
        full_cmd = LeanEnv._build_cmd(state, code)
        result = verify_lean_code(full_cmd, timeout=self.lean_env.timeout)

        original_cmd = full_cmd
        patched_cmd = full_cmd

        if result.get("errors"):
            sorrifier = Sorrifier(
                full_cmd, self.sorrifier_verifier,
                max_cycles=self.sorrifier_cycles,
                verify_timeout=self.sorrifier_timeout,
            )
            patched_cmd = sorrifier.fix_code()
            result = verify_lean_code(patched_cmd, timeout=self.lean_env.timeout)

        closed = result.get("complete", False)

        # Tactic = leaf (children are ignored in is_solved); skeleton = decomposition
        children: tuple[ProofState, ...] = ()
        if action_type == "skeleton":
            children = tuple(
                LeanEnv._parse_proof_state(s.get("goal", ""))
                for s in result.get("sorries", [])
            )

        r_env = self.reward_calc.r_env(original_cmd, patched_cmd, result)
        r_dep = 0.0
        if action_type == "skeleton":
            dep = self.lean_env.dep_graph(patched_cmd)
            r_dep = self.reward_calc.r_dep(dep)

        action = Action(action_type, patched_cmd, children)
        self.graph.expand(state, action, r_env=r_env, r_dep=r_dep, closed=closed)

    @staticmethod
    def _make_prompt(state: ProofState, mode: str) -> str:
        return f"[{mode.upper()}]\n{state}"
