from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from gammazero.core import Action, ProofState
from gammazero.env.lean_env import LeanEnv
from gammazero.policy.prompt import SearchPromptBuilder
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import DependencyRewardAssigner, RewardCalculator
from gammazero.search.sorrifier import Sorrifier

from .batch_executor import BatchExecutor, RolloutBudget
from .failure_handler import FailureHandler
from .heuristic import SearchScorer, SimpleHeuristicScorer
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
        scorer: SearchScorer | None = None,
        prompt_builder: SearchPromptBuilder | None = None,
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
        self.scorer = scorer or SimpleHeuristicScorer()
        self.prompt_builder = prompt_builder or SearchPromptBuilder()

        if executor is None:
            self.failure_handler = failure_handler or FailureHandler(lean, sorrifier, reward)
            self.executor = BatchExecutor(lean, self.failure_handler, reward)
        else:
            self.executor = executor
            self.failure_handler = failure_handler
        self.reward_assigner = reward_assigner or DependencyRewardAssigner(lean, reward)
        self.last_search_metadata: dict = {}
        self._skeleton_feedback_by_state: dict[ProofState, list[str]] = {}

    @property
    def max_nodes(self) -> int:
        return self._budget.max_nodes

    @property
    def total_expanded(self) -> int:
        return self._budget.used

    def _empty_search_metadata(self) -> dict:
        return {
            "budget": {
                "max_nodes": self._budget.max_nodes,
                "used_total": 0,
                "used_tactic": 0,
                "used_skeleton_raw": 0,
                "lean_verify_calls": 0,
                "patch_verify_calls": 0,
            },
            "skeleton_pipeline": {
                "requested": 0,
                "raw_verify_success": 0,
                "raw_verify_failed": 0,
                "patch_attempted": 0,
                "patch_scored": 0,
                "patch_failed": 0,
                "feedback_generated": 0,
                "inserted_raw": 0,
                "selected_by_beam": 0,
                "rejected_by_beam": 0,
                "valid_zero_children": 0,
                "children_new": 0,
                "children_duplicate": 0,
            },
            "final_status": {
                "states": {"OPEN": 0, "SOLVED": 0, "FAILED": 0},
                "actions": {
                    "tactic_SOLVED": 0,
                    "tactic_FAILED": 0,
                    "skeleton_OPEN": 0,
                    "skeleton_SOLVED": 0,
                    "skeleton_FAILED": 0,
                },
            },
            "depth_distribution": {
                "states_seen_by_depth": {},
                "states_solved_by_depth": {},
                "queue_at_stop_by_depth": {},
                "max_depth_reached": 0,
            },
            "beam": {
                "states_pruned_global": 0,
                "states_pruned_per_depth": 0,
                "skeletons_selected": 0,
                "skeletons_rejected_by_beam": 0,
            },
        }

    def _finalize_search_metadata(
        self,
        metadata: dict,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
        queue_at_stop: list[tuple[float, ProofState]],
    ) -> None:
        metadata["budget"]["used_total"] = self._budget.used

        for state in graph.all_states():
            status = graph.status(state)
            metadata["final_status"]["states"][status] += 1
            depth = stats[state].depth if state in stats else graph.get_depth(state)
            key = str(depth)
            metadata["depth_distribution"]["states_seen_by_depth"][key] = (
                metadata["depth_distribution"]["states_seen_by_depth"].get(key, 0) + 1
            )
            if status == "SOLVED":
                metadata["depth_distribution"]["states_solved_by_depth"][key] = (
                    metadata["depth_distribution"]["states_solved_by_depth"].get(key, 0) + 1
                )
            metadata["depth_distribution"]["max_depth_reached"] = max(
                metadata["depth_distribution"]["max_depth_reached"],
                depth,
            )

        for action in graph.all_actions():
            status = graph.status(action)
            key = f"{action.action_type}_{status}"
            if key in metadata["final_status"]["actions"]:
                metadata["final_status"]["actions"][key] += 1

        for _, state in queue_at_stop:
            depth = stats[state].depth if state in stats else graph.get_depth(state)
            key = str(depth)
            metadata["depth_distribution"]["queue_at_stop_by_depth"][key] = (
                metadata["depth_distribution"]["queue_at_stop_by_depth"].get(key, 0) + 1
            )

    def rollout(
        self, theorem: ProofState
    ) -> tuple[list[tuple[ProofState, Action, float, float]], ANDORGraph, dict[Action, float]]:
        self._budget.reset()
        self._skeleton_feedback_by_state = {}
        graph = ANDORGraph(theorem)
        stats: dict[ProofState, StateStats] = {theorem: StateStats(depth=0)}
        queue = StatePriorityQueue()
        seen_states = {self.state_key(theorem)}
        metadata = self._empty_search_metadata()
        stop_reason = "unknown"

        root_score = self.scorer.score_state(theorem, graph, stats)
        stats[theorem].last_score = root_score
        queue.push(theorem, root_score)

        while self._budget.used < self._budget.max_nodes:
            if graph.status(theorem) == "SOLVED":
                stop_reason = "root_solved"
                break

            states = self.pop_state_batch(queue, graph, stats)
            if not states:
                stop_reason = "queue_empty"
                break

            tactic_jobs = self.make_tactic_jobs(states, graph, stats)
            if tactic_jobs:
                actual_counts, action_stats = self.run_jobs(graph, tactic_jobs, "tactic")
                metadata["budget"]["used_tactic"] += action_stats["budget_used"]
                metadata["budget"]["lean_verify_calls"] += action_stats["budget_used"]
                metadata["budget"]["patch_verify_calls"] += action_stats["feedback_generated"]
                for state, count in actual_counts.items():
                    stats[state].tactic_tries += count
                self.update_best_tactic_r_env(graph, action_stats["new_actions"], stats)
                self.propagate(graph, stats)
                if graph.status(theorem) == "SOLVED":
                    stop_reason = "root_solved"
                    break

            skeleton_jobs = self.make_skeleton_jobs(states, graph, stats)
            if skeleton_jobs:
                metadata["skeleton_pipeline"]["requested"] += sum(k for _, k in skeleton_jobs)
                before = set(graph.all_actions())
                actual_counts, action_stats = self.run_jobs(graph, skeleton_jobs, "skeleton")
                metadata["budget"]["used_skeleton_raw"] += action_stats["budget_used"]
                metadata["budget"]["lean_verify_calls"] += action_stats["budget_used"]
                metadata["budget"]["patch_verify_calls"] += action_stats["feedback_generated"]
                metadata["skeleton_pipeline"]["raw_verify_success"] += action_stats["raw_success"]
                metadata["skeleton_pipeline"]["raw_verify_failed"] += action_stats["raw_failed"]
                metadata["skeleton_pipeline"]["patch_attempted"] += action_stats["raw_failed"]
                metadata["skeleton_pipeline"]["patch_scored"] += action_stats["feedback_generated"]
                metadata["skeleton_pipeline"]["patch_failed"] += max(
                    0, action_stats["raw_failed"] - action_stats["feedback_generated"]
                )
                metadata["skeleton_pipeline"]["feedback_generated"] += action_stats["feedback_generated"]
                metadata["skeleton_pipeline"]["inserted_raw"] += action_stats["inserted_raw"]
                for state, count in actual_counts.items():
                    stats[state].skeleton_tries += count
                new_actions = [a for a in graph.all_actions() if a not in before]
                valid_skeletons = self.valid_skeletons_from_actions(graph, new_actions)
                metadata["skeleton_pipeline"]["valid_zero_children"] += sum(
                    1 for action in new_actions
                    if action.action_type == "skeleton"
                    and not self.is_sorrified_action(action)
                    and graph.status(action) != "FAILED"
                    and len(action.children) == 0
                )
                selected = self.select_skeletons(valid_skeletons, graph, stats)
                metadata["skeleton_pipeline"]["selected_by_beam"] += len(selected)
                metadata["skeleton_pipeline"]["rejected_by_beam"] += max(0, len(valid_skeletons) - len(selected))
                metadata["beam"]["skeletons_selected"] += len(selected)
                metadata["beam"]["skeletons_rejected_by_beam"] += max(0, len(valid_skeletons) - len(selected))
                child_stats = self.activate_skeleton_children(selected, graph, stats, queue, seen_states)
                self.update_skeleton_progress(selected, child_stats["generated_by_parent"], stats)
                metadata["skeleton_pipeline"]["children_new"] += child_stats["generated"]
                metadata["skeleton_pipeline"]["children_duplicate"] += child_stats["duplicates"]
                self.propagate(graph, stats)
                if graph.status(theorem) == "SOLVED":
                    stop_reason = "root_solved"
                    break

            for state in states:
                if self.can_requeue_state(state, graph, stats):
                    score = self.scorer.score_state(state, graph, stats)
                    stats[state].last_score = score
                    queue.push(state, score)
                else:
                    self.maybe_exhaust_state(state, graph, stats)

            prune_stats = self.prune_queue(queue, graph, stats)
            metadata["beam"]["states_pruned_global"] += prune_stats["global"]
            metadata["beam"]["states_pruned_per_depth"] += prune_stats["per_depth"]

        else:
            stop_reason = "budget_exhausted"

        queue_at_stop = queue.items()
        self.finalize_unresolved(graph, stats)
        self.propagate(graph, stats)
        self.reward_assigner.assign(graph)
        self.propagate(graph, stats)
        self._finalize_search_metadata(metadata, graph, stats, queue_at_stop)
        self.last_search_metadata = metadata
        graph.search_metadata = metadata

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
        if st.skeleton_exhausted:
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
    ) -> tuple[dict[ProofState, int], dict[str, int]]:
        before = set(graph.all_actions())
        budget_before = self._budget.used
        groups: dict[int, list[ProofState]] = defaultdict(list)
        for state, k in jobs:
            groups[k].append(state)
        for k, states in groups.items():
            if self.total_expanded >= self.max_nodes:
                break
            prompts = [
                self.prompt_builder.build(
                    state,
                    action_type,
                    skeleton_feedbacks=self._skeleton_feedback_by_state.get(state, []),
                )
                for state in states
            ]
            batches = self.policy.sample(states, action_type, k, prompts=prompts)
            feedbacks = self.executor.execute(graph, states, batches, action_type, self._budget, prompts=prompts)
            if action_type == "skeleton":
                self.record_skeleton_feedback(states, feedbacks)
        counts: dict[ProofState, int] = defaultdict(int)
        action_stats = {
            "budget_used": self._budget.used - budget_before,
            "inserted_raw": 0,
            "raw_success": 0,
            "raw_failed": 0,
            "feedback_generated": sum(1 for rows in feedbacks for row in rows if row is not None),
            "new_actions": [],
        }
        for action in graph.all_actions():
            if action in before or action.action_type != action_type:
                continue
            action_stats["new_actions"].append(action)
            parent = graph.get_parent(action)
            if parent is not None:
                counts[parent] += 1
            action_stats["inserted_raw"] += 1
            if graph.status(action) == "FAILED":
                action_stats["raw_failed"] += 1
            else:
                action_stats["raw_success"] += 1
        return dict(counts), action_stats

    def is_sorrified_action(self, action: Action) -> bool:
        return (action.prompt or "").startswith("[SYNTHETIC_PATCH]")

    def record_skeleton_feedback(
        self,
        states: list[ProofState],
        feedbacks: list[list[tuple[str, str, str] | None]],
    ) -> None:
        for state, rows in zip(states, feedbacks):
            for row in rows:
                if row is None:
                    continue
                lean_code, lean_feedback, _ = row
                if not lean_feedback:
                    continue
                block = self.prompt_builder.format_skeleton_feedback(lean_code, lean_feedback)
                self._skeleton_feedback_by_state.setdefault(state, []).append(block)

    def update_best_tactic_r_env(
        self,
        graph: ANDORGraph,
        actions: list[Action],
        stats: dict[ProofState, StateStats],
    ) -> None:
        for action in actions:
            if action.action_type != "tactic":
                continue
            parent = graph.get_parent(action)
            if parent is None or parent not in stats:
                continue
            stats[parent].best_tactic_r_env = max(
                stats[parent].best_tactic_r_env,
                graph.get_r_env(action),
            )

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
            score = self.scorer.score_skeleton(action, parent_state, graph, stats)
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
    ) -> dict[str, int]:
        generated = 0
        duplicates = 0
        generated_by_parent: dict[ProofState, int] = defaultdict(int)
        for parent_state, action, _ in selected_skeletons:
            parent_depth = stats[parent_state].depth
            for child in action.children:
                key = self.state_key(child)
                if key in seen_states:
                    duplicates += 1
                    continue
                seen_states.add(key)
                depth = parent_depth + 1
                skeleton_score = self.scorer.score_skeleton(action, parent_state, graph, stats)
                skeleton_r_env = graph.get_r_env(action)
                stats[child] = StateStats(
                    depth=depth,
                    incoming_skeleton_score=skeleton_score,
                    incoming_skeleton_r_env=skeleton_r_env,
                )
                graph.add_state(child, depth=depth)
                score = self.scorer.score_state(child, graph, stats)
                stats[child].last_score = score
                queue.push(child, score)
                generated += 1
                generated_by_parent[parent_state] += 1
        return {
            "generated": generated,
            "duplicates": duplicates,
            "generated_by_parent": dict(generated_by_parent),
        }

    def update_skeleton_progress(
        self,
        selected_skeletons: list[tuple[ProofState, Action, float]],
        generated_by_parent: dict[ProofState, int],
        stats: dict[ProofState, StateStats],
    ) -> None:
        touched_parents = {parent_state for parent_state, _, _ in selected_skeletons}
        for parent_state in touched_parents:
            st = stats[parent_state]
            new_children = generated_by_parent.get(parent_state, 0)
            st.last_skeleton_new_children = new_children
            if new_children == 0:
                st.bad_skeleton_rounds += 1
            else:
                st.bad_skeleton_rounds = 0
            if st.bad_skeleton_rounds >= 1:
                st.skeleton_exhausted = True

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
    ) -> dict[str, int]:
        rows = []
        while len(queue) > 0:
            state, _ = queue.pop()
            if state is None:
                break
            if graph.status(state) != "OPEN" or stats[state].exhausted:
                continue
            score = self.scorer.score_state(state, graph, stats)
            stats[state].last_score = score
            rows.append((state, score))

        rows.sort(key=lambda x: x[1], reverse=True)
        before_global = len(rows)
        rows = rows[: self.state_beam_width]
        global_pruned = before_global - len(rows)

        per_depth: dict[int, list[tuple[ProofState, float]]] = defaultdict(list)
        for state, score in rows:
            per_depth[stats[state].depth].append((state, score))

        kept = []
        per_depth_pruned = 0
        for xs in per_depth.values():
            xs.sort(key=lambda x: x[1], reverse=True)
            kept_xs = xs[: self.state_beam_per_depth]
            per_depth_pruned += len(xs) - len(kept_xs)
            kept.extend(kept_xs)
        queue.rebuild(kept)
        return {"global": global_pruned, "per_depth": per_depth_pruned}

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
