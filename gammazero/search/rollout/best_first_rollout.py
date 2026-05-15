from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from gammazero.core import Action, ProofState
from gammazero.env.lean_env import LeanEnv
from gammazero.policy.prompt import build_prompt
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import DependencyRewardAssigner, RewardCalculator
from gammazero.search.sorrifier import Sorrifier

from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler
from .heuristic import DefaultScorer, HeuristicScorer
from .search_queue import StatePriorityQueue
from .search_stats import StateStats


class SamplePolicy(Protocol):
    def sample(
        self, states: list[ProofState], action_type: str, n: int, *, prompts: list[str] | None = None
    ) -> list[list[dict]]: ...


class BestFirstRollout:
    """Best-first progressive rollout over the existing AND/OR proof graph."""

    def __init__(
        self,
        policy: SamplePolicy,
        lean: LeanEnv,
        sorrifier: Sorrifier,
        reward: RewardCalculator,
        *,
        max_depth: int,
        max_nodes: int,
        search_batch_size: int = 4,
        initial_tactic_k: int = 4,
        retry_tactic_k: int = 4,
        max_tactic_per_state: int = 16,
        initial_skeleton_k: int = 4,
        retry_skeleton_k: int = 2,
        max_skeleton_per_state: int = 8,
        state_beam_width: int = 32,
        state_beam_per_depth: int = 8,
        skeleton_beam_per_state: int = 2,
        scorer: HeuristicScorer | None = None,
        executor: BatchExecutor | None = None,
        failure_handler: FailureHandler | None = None,
        reward_assigner: DependencyRewardAssigner | None = None,
    ):
        self.policy = policy
        self.lean = lean
        self.sorrifier = sorrifier
        self.reward = reward

        self.max_depth = max_depth
        self._budget = RolloutBudget(max_nodes)

        self.search_batch_size = search_batch_size
        self.initial_tactic_k = initial_tactic_k
        self.retry_tactic_k = retry_tactic_k
        self.max_tactic_per_state = max_tactic_per_state
        self.initial_skeleton_k = initial_skeleton_k
        self.retry_skeleton_k = retry_skeleton_k
        self.max_skeleton_per_state = max_skeleton_per_state
        self.state_beam_width = state_beam_width
        self.state_beam_per_depth = state_beam_per_depth
        self.skeleton_beam_per_state = skeleton_beam_per_state
        self.scorer = scorer or DefaultScorer()

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

    def H_state(self, state: ProofState, graph: ANDORGraph, stats: dict[ProofState, StateStats]) -> float:
        return self.scorer.score_state(state, graph, stats)

    def H_skeleton(
        self,
        action: Action,
        parent_state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        return self.scorer.score_skeleton(action, parent_state, graph, stats)

    def rollout(
        self, theorem: ProofState
    ) -> tuple[list[tuple[ProofState, Action, float, float]], ANDORGraph, dict[Action, float]]:
        self._budget.reset()
        graph = ANDORGraph(theorem)
        stats: dict[ProofState, StateStats] = {theorem: StateStats(depth=0)}
        queue = StatePriorityQueue()
        seen_states = {self.state_key(theorem)}

        root_score = self.H_state(theorem, graph, stats)
        stats[theorem].last_score = root_score
        queue.push(theorem, root_score)

        while self._budget.used < self._budget.max_nodes:
            if graph.status(theorem) == "SOLVED":
                break

            states = self.pop_state_batch(queue, graph, stats)
            if not states:
                break

            tactic_jobs = self.make_tactic_jobs(states, graph, stats)
            if tactic_jobs:
                actual_counts = self.run_jobs(graph, tactic_jobs, "tactic")
                for state, count in actual_counts.items():
                    stats[state].tactic_tries += count
                self.propagate(graph, stats)
                if graph.status(theorem) == "SOLVED":
                    break

            skeleton_jobs = self.make_skeleton_jobs(states, graph, stats)
            if skeleton_jobs:
                before = set(graph.all_actions())
                actual_counts = self.run_jobs(graph, skeleton_jobs, "skeleton")
                for state, count in actual_counts.items():
                    stats[state].skeleton_tries += count
                new_actions = [a for a in graph.all_actions() if a not in before]
                valid_skeletons = self.valid_skeletons_from_actions(graph, new_actions)
                selected = self.select_skeletons(valid_skeletons, graph, stats)
                self.activate_skeleton_children(selected, graph, stats, queue, seen_states)
                self.propagate(graph, stats)
                if graph.status(theorem) == "SOLVED":
                    break

            for state in states:
                if self.can_requeue_state(state, graph, stats):
                    score = self.H_state(state, graph, stats)
                    stats[state].last_score = score
                    queue.push(state, score)
                else:
                    self.maybe_exhaust_state(state, graph, stats)

            self.prune_queue(queue, graph, stats)

        self.finalize_unresolved(graph, stats)
        self.propagate(graph, stats)

        self.reward_assigner.assign(graph)
        q_values = self.reward.compute_returns(graph)
        samples: list[tuple[ProofState, Action, float, float]] = []
        for action, q in q_values.items():
            parent = graph.get_parent(action, theorem)
            samples.append((parent, action, graph.get_r_env(action), q))
        return samples, graph, q_values

    def pop_state_batch(
        self,
        queue: StatePriorityQueue,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> list[ProofState]:
        out: list[ProofState] = []
        while len(out) < self.search_batch_size and len(queue) > 0:
            state, _ = queue.pop()
            if state is None:
                break
            if graph.status(state) != "OPEN":
                continue
            if stats[state].exhausted:
                continue
            if stats[state].depth > self.max_depth:
                continue
            out.append(state)
        return out

    def should_try_tactic(
        self, state: ProofState, graph: ANDORGraph, stats: dict[ProofState, StateStats]
    ) -> bool:
        if graph.status(state) != "OPEN":
            return False
        st = stats[state]
        if st.exhausted or st.depth > self.max_depth:
            return False
        if st.tactic_tries >= self.max_tactic_per_state:
            return False
        return True

    def tactic_width(self, state: ProofState, stats: dict[ProofState, StateStats]) -> int:
        st = stats[state]
        k = self.initial_tactic_k if st.tactic_tries == 0 else self.retry_tactic_k
        remaining = self.max_tactic_per_state - st.tactic_tries
        remaining_budget = self.max_nodes - self.total_expanded
        return max(0, min(k, remaining, remaining_budget))

    def make_tactic_jobs(
        self,
        states: list[ProofState],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> list[tuple[ProofState, int]]:
        jobs = []
        for state in states:
            if not self.should_try_tactic(state, graph, stats):
                continue
            k = self.tactic_width(state, stats)
            if k <= 0:
                continue
            jobs.append((state, k))
            stats[state].tactic_probe_done = True
        return jobs

    def should_expand_skeleton(
        self, state: ProofState, graph: ANDORGraph, stats: dict[ProofState, StateStats]
    ) -> bool:
        if graph.status(state) != "OPEN":
            return False
        st = stats[state]
        if st.exhausted or st.depth >= self.max_depth:
            return False
        if st.skeleton_tries >= self.max_skeleton_per_state:
            return False
        if not st.tactic_probe_done:
            return False
        return True

    def skeleton_width(self, state: ProofState, stats: dict[ProofState, StateStats]) -> int:
        st = stats[state]
        k = self.initial_skeleton_k if st.skeleton_tries == 0 else self.retry_skeleton_k
        remaining = self.max_skeleton_per_state - st.skeleton_tries
        remaining_budget = self.max_nodes - self.total_expanded
        return max(0, min(k, remaining, remaining_budget))

    def make_skeleton_jobs(
        self,
        states: list[ProofState],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> list[tuple[ProofState, int]]:
        jobs = []
        for state in states:
            if not self.should_expand_skeleton(state, graph, stats):
                continue
            k = self.skeleton_width(state, stats)
            if k <= 0:
                continue
            jobs.append((state, k))
            stats[state].skeleton_probe_done = True
        return jobs

    def run_jobs(
        self,
        graph: ANDORGraph,
        jobs: list[tuple[ProofState, int]],
        action_type: str,
    ) -> dict[ProofState, int]:
        before = set(graph.all_actions())
        groups: dict[int, list[ProofState]] = defaultdict(list)
        for state, k in jobs:
            groups[k].append(state)
        for k, states in groups.items():
            if self.total_expanded >= self.max_nodes:
                break
            prompts = [build_prompt(state, action_type) for state in states]
            batches = self.policy.sample(states, action_type, k, prompts=prompts)
            self.executor.execute(graph, states, batches, action_type, self._budget, prompts=prompts)
        counts: dict[ProofState, int] = defaultdict(int)
        for action in graph.all_actions():
            if action in before or action.action_type != action_type:
                continue
            parent = graph.get_parent(action)
            if parent is not None:
                counts[parent] += 1
        return dict(counts)

    def valid_skeletons_from_actions(
        self, graph: ANDORGraph, actions: list[Action]
    ) -> list[tuple[ProofState, Action]]:
        valid = []
        for action in actions:
            if action.action_type != "skeleton":
                continue
            parent = graph.get_parent(action)
            if parent is None:
                continue
            if graph.status(action) == "OPEN" and len(action.children) > 0:
                valid.append((parent, action))
            elif graph.status(action) != "SOLVED":
                graph.mark_failed(action)
        return valid

    def select_skeletons(
        self,
        valid_skeletons: list[tuple[ProofState, Action]],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> list[tuple[ProofState, Action, float]]:
        by_parent: dict[ProofState, list[tuple[float, ProofState, Action]]] = defaultdict(list)
        for parent_state, action in valid_skeletons:
            score = self.H_skeleton(action, parent_state, graph, stats)
            by_parent[parent_state].append((score, parent_state, action))

        selected = []
        for rows in by_parent.values():
            rows.sort(key=lambda x: x[0], reverse=True)
            for score, parent_state, action in rows[: self.skeleton_beam_per_state]:
                selected.append((parent_state, action, score))
            for _, _, action in rows[self.skeleton_beam_per_state :]:
                graph.mark_failed(action)
        return selected

    def activate_skeleton_children(
        self,
        selected_skeletons: list[tuple[ProofState, Action, float]],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
        queue: StatePriorityQueue,
        seen_states: set[str],
    ) -> None:
        for parent_state, action, _ in selected_skeletons:
            parent_depth = stats[parent_state].depth
            for child in action.children:
                key = self.state_key(child)
                if key in seen_states:
                    continue
                seen_states.add(key)
                depth = parent_depth + 1
                stats[child] = StateStats(depth=depth)
                graph.add_state(child, depth=depth)
                score = self.H_state(child, graph, stats)
                stats[child].last_score = score
                queue.push(child, score)

    def state_key(self, state: ProofState) -> str:
        ctx = " ".join((state.context or "").split())
        goal = " ".join((state.goal or "").split())
        return ctx + "\n⊢ " + goal

    def can_requeue_state(
        self, state: ProofState, graph: ANDORGraph, stats: dict[ProofState, StateStats]
    ) -> bool:
        if graph.status(state) != "OPEN":
            return False
        st = stats[state]
        if st.exhausted or st.depth >= self.max_depth:
            return False
        return st.tactic_tries < self.max_tactic_per_state or st.skeleton_tries < self.max_skeleton_per_state

    def maybe_exhaust_state(
        self, state: ProofState, graph: ANDORGraph, stats: dict[ProofState, StateStats]
    ) -> None:
        if graph.status(state) != "OPEN":
            return
        st = stats[state]
        if st.depth >= self.max_depth:
            st.exhausted = True
        elif st.tactic_tries >= self.max_tactic_per_state and st.skeleton_tries >= self.max_skeleton_per_state:
            st.exhausted = True
        else:
            return
        actions = graph.get_actions(state)
        if actions and all(graph.status(action) == "FAILED" for action in actions):
            graph.mark_failed(state)

    def prune_queue(
        self,
        queue: StatePriorityQueue,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> None:
        rows = []
        while len(queue) > 0:
            state, score = queue.pop()
            if state is None or score is None:
                break
            if graph.status(state) != "OPEN" or stats[state].exhausted:
                continue
            rows.append((state, score))

        rows.sort(key=lambda x: x[1], reverse=True)
        rows = rows[: self.state_beam_width]

        per_depth: dict[int, list[tuple[ProofState, float]]] = defaultdict(list)
        for state, score in rows:
            per_depth[stats[state].depth].append((state, score))

        kept = []
        for xs in per_depth.values():
            xs.sort(key=lambda x: x[1], reverse=True)
            kept.extend(xs[: self.state_beam_per_depth])
        queue.rebuild(kept)

    def propagate(self, graph: ANDORGraph, stats: dict[ProofState, StateStats]) -> None:
        changed = True
        while changed:
            changed = False
            for action in graph.all_actions():
                if graph.status(action) != "OPEN" or action.action_type != "skeleton":
                    continue
                if not action.children:
                    changed |= graph.mark_failed(action)
                elif all(graph.status(child) == "SOLVED" for child in action.children):
                    changed |= graph.mark_solved(action)
                elif any(graph.status(child) == "FAILED" for child in action.children):
                    changed |= graph.mark_failed(action)

            for state in graph.all_states():
                if graph.status(state) != "OPEN":
                    continue
                actions = graph.get_actions(state)
                if any(graph.status(action) == "SOLVED" for action in actions):
                    changed |= graph.mark_solved(state)
                elif stats.get(state) and stats[state].exhausted:
                    if actions and all(graph.status(action) == "FAILED" for action in actions):
                        changed |= graph.mark_failed(state)

    def finalize_unresolved(self, graph: ANDORGraph, stats: dict[ProofState, StateStats]) -> None:
        for state in graph.all_states():
            if graph.status(state) != "OPEN":
                continue
            if state in stats:
                stats[state].exhausted = True
            graph.mark_failed(state)

        for action in graph.all_actions():
            if graph.status(action) == "OPEN":
                graph.mark_failed(action)
