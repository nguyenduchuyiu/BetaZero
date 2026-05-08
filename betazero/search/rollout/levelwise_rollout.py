from __future__ import annotations

from typing import Protocol

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.prompt import build_prompt
from betazero.search.graph import ANDORGraph
from betazero.search.reward import DependencyRewardAssigner, RewardCalculator
from betazero.search.sorrifier import Sorrifier

from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler


class SamplePolicy(Protocol):
    """`n` = completions per state; return[i] has up to `n` strings for states[i]."""

    def sample(
        self, states: list[ProofState], action_type: str, n: int, *, prompts: list[str] | None = None
    ) -> list[list[dict]]: ...


class LevelwiseRollout:
    """Runs level-wise tactic and skeleton rollout over the proof graph under a node budget."""

    def __init__(
        self,
        policy: SamplePolicy,
        lean: LeanEnv,
        sorrifier: Sorrifier,
        reward: RewardCalculator,
        K: int = 32,
        max_depth: int = 5,
        max_nodes: int = 128,
        tactic_ratio: float = 0.8,
        *,
        executor: BatchExecutor | None = None,
        failure_handler: FailureHandler | None = None,
        reward_assigner: DependencyRewardAssigner | None = None,
    ):
        assert K >= 1, "K must be at least 1"
        self.policy = policy
        self.lean = lean
        self.reward = reward
        self.K_tac = max(1, int(K * tactic_ratio))
        self.K_skel = K - self.K_tac
        self.max_depth = max_depth
        self._budget = RolloutBudget(max_nodes)

        if executor is None:
            self.failure_handler = failure_handler or FailureHandler(lean, sorrifier, reward)
            self.executor = BatchExecutor(lean, self.failure_handler, reward)
        else:
            self.executor = executor
            self.failure_handler = failure_handler
            
        self.reward_assigner = reward_assigner or DependencyRewardAssigner(lean, reward)

    @property
    def max_nodes(self) -> int:
        return self._budget.max_nodes

    @property
    def total_expanded(self) -> int:
        return self._budget.used

    def rollout(
        self, theorem: ProofState
    ) -> tuple[
        list[tuple[ProofState, Action, float, float]],
        ANDORGraph,
        dict[Action, float],
    ]:
        self._budget.reset()
        graph = ANDORGraph(theorem)
        for depth in range(self.max_depth):
            if graph.is_solved(theorem):
                break

            frontier = [s for s in graph.unsolved_states() if graph.get_depth(s) == depth]
            if not frontier or self._budget.used >= self._budget.max_nodes:
                break

            self._run_tactic_phase(graph, frontier)

            # Re-check solved status and budget before skeleton phase
            if graph.is_solved(theorem) or self._budget.used >= self._budget.max_nodes:
                break

            skel_frontier = [s for s in frontier if not graph.is_solved(s)]
            if skel_frontier and self.K_skel > 0:
                self._run_skeleton_phase(graph, skel_frontier)

        self.reward_assigner.assign(graph)
        q_values = self.reward.compute_returns(graph)
        samples: list[tuple[ProofState, Action, float, float]] = []
        for a, q in q_values.items():
            tup = (graph.get_parent(a, theorem), a, graph.get_r_env(a), q)
            samples.append(tup)
        return samples, graph, q_values

    def _run_tactic_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        """
        Executes a single-stage tactic rollout.
        """
        # BUDGET CHECK: Only process as many states as we can afford
        remaining = self.max_nodes - self.total_expanded
        if remaining <= 0:
            return
        
        # Limit frontier to what we can actually execute
        max_states = max(1, remaining // self.K_tac)
        active_frontier = frontier[:max_states]

        first_prompts = [build_prompt(s, "tactic") for s in active_frontier]
        first_round_actions = self.policy.sample(
            active_frontier, "tactic", self.K_tac, prompts=first_prompts
        )
        self.executor.execute(
            graph, active_frontier, first_round_actions, "tactic", self._budget, prompts=first_prompts
        )

    def _run_skeleton_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        # BUDGET CHECK: Only process as many states as we can afford
        remaining = self.max_nodes - self.total_expanded
        if remaining <= 0:
            return
            
        max_states = max(1, remaining // self.K_skel)
        active_frontier = frontier[:max_states]

        skel_prompts = [build_prompt(s, "skeleton") for s in active_frontier]
        skel_batches = self.policy.sample(active_frontier, "skeleton", self.K_skel, prompts=skel_prompts)
        self.executor.execute(
            graph, active_frontier, skel_batches, "skeleton", self._budget, prompts=skel_prompts
        )
