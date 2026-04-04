from __future__ import annotations

from typing import Protocol

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.prompt import build_tactic_self_correct_prompt
from betazero.search import ANDORGraph
from betazero.search import RewardCalculator
from betazero.search import DependencyRewardAssigner
from betazero.search import Sorrifier

from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler


class SamplePolicy(Protocol):
    """`n` = completions per state; return[i] has up to `n` strings for states[i]."""

    def sample(
        self, states: list[ProofState], action_type: str, n: int, *, prompts: list[str] | None = None
    ) -> list[list[str]]: ...


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

    def rollout(self, theorem: ProofState) -> list[tuple[ProofState, Action, float, float]]:
        graph = ANDORGraph(theorem)

        for depth in range(self.max_depth):
            frontier = [s for s in graph.unsolved_states() if graph.get_depth(s) == depth]
            if not frontier or self._budget.used >= self._budget.max_nodes:
                break

            self._run_tactic_phase(graph, frontier)

            skel_frontier = [s for s in frontier if not graph.is_solved(s)]
            if skel_frontier and self.K_skel > 0 and self._budget.used < self._budget.max_nodes:
                self._run_skeleton_phase(graph, skel_frontier)

        self.reward_assigner.assign(graph)
        q_values = self.reward.compute_returns(graph)
        return [(graph.get_parent(a, theorem), a, graph.get_r_env(a), q) for a, q in q_values.items()]

    def _run_tactic_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        """
        Executes a two-stage tactic rollout with a self-correction loop.
        First attempt is exploratory; the second attempt uses feedback from 
        failed attempts to refine the policy's output.
        """
        # Split the tactic budget into two rounds
        first_round_budget = max(1, self.K_tac // 2)
        second_round_budget = self.K_tac - first_round_budget

        # Stage 1: Initial exploratory sampling
        first_round_actions = self.policy.sample(frontier, "tactic", first_round_budget)
        # Execute in parallel and collect feedback (error msg, patched code) for potential correction
        round_one_outcomes = self.executor.execute(
            graph, frontier, first_round_actions, "tactic", self._budget
        )

        # Prepare self-correction data for states that were not solved in Round 1
        correction_states: list[ProofState] = []
        correction_prompts: list[str] = []
        
        for state, per_action in zip(frontier, round_one_outcomes):
            if graph.is_solved(state):
                continue
            for feedback in per_action:
                if feedback is None:
                    continue
                correction_states.append(state)
                correction_prompts.append(build_tactic_self_correct_prompt(state, *feedback))

        # Stage 2: Self-correction attempt using refined prompts
        if correction_states and self._budget.used < self._budget.max_nodes:
            second_round_actions = self.policy.sample(
                correction_states, "tactic", second_round_budget, prompts=correction_prompts
            )
            # Verify the corrected actions in parallel
            self.executor.execute(graph, correction_states, second_round_actions, "tactic", self._budget)

    def _run_skeleton_phase(self, graph: ANDORGraph, frontier: list[ProofState]) -> None:
        skel_batches = self.policy.sample(frontier, "skeleton", self.K_skel)
        self.executor.execute(graph, frontier, skel_batches, "skeleton", self._budget)
