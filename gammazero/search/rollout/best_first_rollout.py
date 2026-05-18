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
        min_tactic_before_skeleton: int = 8,
        promising_tactic_r_env: float = 0.4,
        strong_tactic_r_env: float = 0.7,
        initial_skeleton_k: int = 4,
        retry_skeleton_k: int = 2,
        max_skeleton_per_state: int = 8,
        state_beam_width: int = 32,
        state_beam_per_depth: int = 8,
        skeleton_beam_per_state: int = 2,
        skeleton_commitment: bool = True,
        max_reserved_skeletons_per_state: int = 2,
        commit_stale_rounds_before_fallback: int = 2,
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
        self.min_tactic_before_skeleton = min_tactic_before_skeleton
        self.promising_tactic_r_env = promising_tactic_r_env
        self.strong_tactic_r_env = strong_tactic_r_env
        self.initial_skeleton_k = initial_skeleton_k
        self.retry_skeleton_k = retry_skeleton_k
        self.max_skeleton_per_state = max_skeleton_per_state
        self.state_beam_width = state_beam_width
        self.state_beam_per_depth = state_beam_per_depth
        self.skeleton_beam_per_state = skeleton_beam_per_state
        self.skeleton_commitment = skeleton_commitment
        self.max_reserved_skeletons_per_state = max_reserved_skeletons_per_state
        self.commit_stale_rounds_before_fallback = commit_stale_rounds_before_fallback
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
        self._tactic_feedback_by_state: dict[ProofState, list[str]] = {}
        self._skeleton_feedback_by_state: dict[ProofState, list[str]] = {}
        self._blocked_new_skeleton_due_to_active_commit = 0
        self._duplicate_skeleton_actions = 0

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
                "skeleton_duplicate_actions": 0,
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
            "skeleton_commitment": {
                "committed": 0,
                "reserved": 0,
                "fallback_activated": 0,
                "committed_solved": 0,
                "committed_failed": 0,
                "committed_stale": 0,
                "blocked_new_skeleton_due_to_active_commit": 0,
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

        committed_solved = 0
        committed_failed = 0
        for st in stats.values():
            if st.committed_skeleton is not None:
                status = graph.status(st.committed_skeleton)
                if status == "SOLVED":
                    committed_solved += 1
                elif status == "FAILED":
                    committed_failed += 1
            committed_failed += st.skeleton_commit_failed_count
        metadata["skeleton_commitment"]["committed_solved"] = committed_solved
        metadata["skeleton_commitment"]["committed_failed"] = committed_failed

    def rollout(
        self, theorem: ProofState
    ) -> tuple[list[tuple[ProofState, Action, float, float]], ANDORGraph, dict[Action, float]]:
        self._budget.reset()
        self._tactic_feedback_by_state = {}
        self._skeleton_feedback_by_state = {}
        self._blocked_new_skeleton_due_to_active_commit = 0
        graph = ANDORGraph(theorem)
        stats: dict[ProofState, StateStats] = {theorem: StateStats(depth=0)}
        queue = StatePriorityQueue()
        seen_states = {self.state_key(theorem): theorem}
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

            if self.skeleton_commitment:
                refresh_stats = self.refresh_commitments(graph, stats, queue, seen_states)
                self.record_commitment_refresh(metadata, refresh_stats)
                if refresh_stats["fallback_activated"]:
                    self.propagate(graph, stats)
                    if graph.status(theorem) == "SOLVED":
                        stop_reason = "root_solved"
                        break

            blocked_before = self._blocked_new_skeleton_due_to_active_commit
            skeleton_jobs = self.make_skeleton_jobs(states, graph, stats)
            metadata["skeleton_commitment"]["blocked_new_skeleton_due_to_active_commit"] += (
                self._blocked_new_skeleton_due_to_active_commit - blocked_before
            )
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
                duplicate_skeletons_before = self._duplicate_skeleton_actions
                valid_skeletons = self.valid_skeletons_from_actions(graph, new_actions)
                metadata["skeleton_pipeline"]["skeleton_duplicate_actions"] += (
                    self._duplicate_skeleton_actions - duplicate_skeletons_before
                )
                metadata["skeleton_pipeline"]["valid_zero_children"] += sum(
                    1 for action in new_actions
                    if action.action_type == "skeleton"
                    and not self.is_sorrified_action(action)
                    and graph.status(action) != "FAILED"
                    and len(action.children) == 0
                )
                selected = self.select_skeletons(valid_skeletons, graph, stats)
                if self.skeleton_commitment:
                    metadata["skeleton_commitment"]["committed"] += len(selected)
                    metadata["skeleton_commitment"]["reserved"] += max(
                        0, len(valid_skeletons) - len(selected)
                    )
                metadata["skeleton_pipeline"]["selected_by_beam"] += len(selected)
                rejected = 0 if self.skeleton_commitment else max(0, len(valid_skeletons) - len(selected))
                metadata["skeleton_pipeline"]["rejected_by_beam"] += rejected
                metadata["beam"]["skeletons_selected"] += len(selected)
                metadata["beam"]["skeletons_rejected_by_beam"] += rejected
                child_stats = self.activate_skeleton_children(selected, graph, stats, queue, seen_states)
                self.update_skeleton_progress(
                    selected,
                    child_stats["generated_by_parent"],
                    stats,
                    child_stats["active_by_parent"],
                )
                metadata["skeleton_pipeline"]["children_new"] += child_stats["generated"]
                metadata["skeleton_pipeline"]["children_duplicate"] += child_stats["duplicates"]
                self.propagate(graph, stats)
                if self.skeleton_commitment:
                    refresh_stats = self.refresh_commitments(graph, stats, queue, seen_states)
                    self.record_commitment_refresh(metadata, refresh_stats)
                    if refresh_stats["fallback_activated"]:
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
        self.reward_assigner.stitch_and_score_skeletons(graph)
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
        if self.skeleton_commitment and st.committed_skeleton is not None:
            self._blocked_new_skeleton_due_to_active_commit += 1
            return False
        if self.skeleton_commitment and st.reserved_skeletons:
            return False
        if st.skeleton_exhausted:
            return False
        if st.skeleton_tries >= self.max_skeleton_per_state:
            return False
        if not st.tactic_probe_done:
            return False
        if st.tactic_tries < self.min_tactic_before_skeleton:
            return False
        if st.best_tactic_r_env >= self.strong_tactic_r_env and st.tactic_tries < self.max_tactic_per_state:
            return False
        if st.best_tactic_r_env >= self.promising_tactic_r_env and st.tactic_tries < self.max_tactic_per_state:
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
            prompts = []
            for state in states:
                if action_type == "tactic":
                    target = BatchExecutor._subgoal_tactic_target(graph, state)
                    if target is not None:
                        parent_state, skeleton, target_child_index = target
                        prompts.append(
                            self.prompt_builder.build_subgoal_tactic(
                                parent_state,
                                skeleton,
                                target_child_index,
                                tactic_feedbacks=self._tactic_feedback_by_state.get(state, []),
                            )
                        )
                        continue
                prompts.append(
                    self.prompt_builder.build(
                        state,
                        action_type,
                        tactic_feedbacks=self._tactic_feedback_by_state.get(state, []),
                        skeleton_feedbacks=self._skeleton_feedback_by_state.get(state, []),
                    )
                )
            batches = self.policy.sample(states, action_type, k, prompts=prompts)
            feedbacks = self.executor.execute(graph, states, batches, action_type, self._budget, prompts=prompts)
            if action_type == "tactic":
                self.record_tactic_feedback(states, feedbacks)
            elif action_type == "skeleton":
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

    def record_tactic_feedback(
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
                block = self.prompt_builder.format_tactic_feedback(lean_code, lean_feedback)
                self._tactic_feedback_by_state.setdefault(state, []).append(block)

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
        action_set = set(actions)
        seen_signatures = {
            sig
            for existing in graph.all_actions()
            if existing not in action_set
            and existing.action_type == "skeleton"
            and len(existing.children) > 0
            for sig in [self.skeleton_signature(graph, existing)]
            if sig is not None
        }
        for action in actions:
            if action.action_type != "skeleton":
                continue
            parent = graph.get_parent(action)
            if parent is None:
                continue
            if graph.status(action) == "OPEN" and len(action.children) > 0:
                signature = self.skeleton_signature(graph, action)
                if signature in seen_signatures:
                    graph.mark_failed(action)
                    self._duplicate_skeleton_actions += 1
                    continue
                seen_signatures.add(signature)
                valid.append((parent, action))
            elif graph.status(action) != "SOLVED":
                graph.mark_failed(action)
        return valid

    def skeleton_signature(
        self,
        graph: ANDORGraph,
        action: Action,
    ) -> tuple[str, tuple[str, ...]] | None:
        parent = graph.get_parent(action)
        if parent is None:
            return None
        return (
            self.state_key(parent),
            tuple(self.state_key(child) for child in action.children),
        )

    def select_skeletons(
        self,
        valid_skeletons: list[tuple[ProofState, Action]],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> list[tuple[ProofState, Action, float]]:
        if not self.skeleton_commitment:
            return self.select_skeletons_beam(valid_skeletons, graph, stats)

        by_parent: dict[ProofState, list[tuple[float, Action]]] = defaultdict(list)
        for parent_state, action in valid_skeletons:
            if graph.status(parent_state) != "OPEN":
                graph.mark_failed(action)
                continue
            score = self.scorer.score_skeleton(action, parent_state, graph, stats)
            parent_stats = stats[parent_state]
            committed = parent_stats.committed_skeleton
            if committed is not None and graph.status(committed) in ("OPEN", "SOLVED"):
                self.reserve_skeleton(graph, parent_stats, score, action)
                continue
            by_parent[parent_state].append((score, action))

        selected = []
        for parent_state, rows in by_parent.items():
            rows.sort(key=lambda x: x[0], reverse=True)
            if not rows:
                continue

            best_score, best_action = rows[0]
            stats[parent_state].committed_skeleton = best_action
            stats[parent_state].committed_skeleton_progress_last = self.count_solved_children(
                best_action, graph
            )
            stats[parent_state].committed_skeleton_stale_rounds = 0
            selected.append((parent_state, best_action, best_score))

            for score, action in rows[1:]:
                self.reserve_skeleton(graph, stats[parent_state], score, action)

        return selected

    def reserve_skeleton(self, graph: ANDORGraph, st: StateStats, score: float, action: Action) -> None:
        graph.mark_failed(action)
        st.reserved_skeletons.append((score, action))
        st.reserved_skeletons.sort(key=lambda x: x[0], reverse=True)
        if self.max_reserved_skeletons_per_state >= 0:
            del st.reserved_skeletons[self.max_reserved_skeletons_per_state :]

    def select_skeletons_beam(
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
        seen_states: dict[str, ProofState] | set[str],
    ) -> dict[str, int]:
        generated = 0
        duplicates = 0
        generated_by_parent: dict[ProofState, int] = defaultdict(int)
        active_by_parent: dict[ProofState, int] = defaultdict(int)
        for parent_state, action, _ in selected_skeletons:
            parent_depth = stats[parent_state].depth
            parent_stats = stats[parent_state]
            for child in action.children:
                key = self.state_key(child)
                if isinstance(seen_states, dict) and key in seen_states:
                    canonical = seen_states[key]
                    duplicates += 1
                    if canonical in stats:
                        if (parent_state, action) not in stats[canonical].parent_skeletons:
                            stats[canonical].parent_skeletons.append((parent_state, action))
                    parent_stats.active_skeleton_children.add(canonical)
                    active_by_parent[parent_state] += 1
                    continue
                if not isinstance(seen_states, dict) and key in seen_states:
                    duplicates += 1
                    parent_stats.active_skeleton_children.add(child)
                    active_by_parent[parent_state] += 1
                    continue

                if isinstance(seen_states, dict):
                    seen_states[key] = child
                else:
                    seen_states.add(key)
                depth = parent_depth + 1
                skeleton_score = self.scorer.score_skeleton(action, parent_state, graph, stats)
                skeleton_r_env = graph.get_r_env(action)
                stats[child] = StateStats(
                    depth=depth,
                    incoming_skeleton_score=skeleton_score,
                    incoming_skeleton_r_env=skeleton_r_env,
                    parent_skeletons=[(parent_state, action)],
                )
                graph.add_state(child, depth=depth)
                parent_stats.active_skeleton_children.add(child)
                score = self.scorer.score_state(child, graph, stats)
                stats[child].last_score = score
                queue.push(child, score)
                generated += 1
                generated_by_parent[parent_state] += 1
                active_by_parent[parent_state] += 1
        return {
            "generated": generated,
            "duplicates": duplicates,
            "generated_by_parent": dict(generated_by_parent),
            "active_by_parent": dict(active_by_parent),
        }

    def try_activate_reserved_skeleton(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
        queue: StatePriorityQueue,
        seen_states: dict[str, ProofState],
    ) -> dict[str, int] | None:
        st = stats[state]
        reserve = [
            (score, action)
            for score, action in st.reserved_skeletons
            if graph.status(action) != "SOLVED"
        ]
        reserve.sort(key=lambda x: x[0], reverse=True)
        st.reserved_skeletons = reserve

        if not reserve:
            return None

        score, action = reserve.pop(0)
        graph.mark_open(action)
        st.committed_skeleton = action
        st.committed_skeleton_progress_last = self.count_solved_children(action, graph)
        st.committed_skeleton_stale_rounds = 0
        child_stats = self.activate_skeleton_children(
            [(state, action, score)],
            graph,
            stats,
            queue,
            seen_states,
        )
        self.update_skeleton_progress(
            [(state, action, score)],
            child_stats["generated_by_parent"],
            stats,
            child_stats["active_by_parent"],
        )
        return child_stats

    def activate_reserved_skeletons_for_failed_commits(
        self,
        states: list[ProofState],
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
        queue: StatePriorityQueue,
        seen_states: dict[str, ProofState],
    ) -> dict[str, int]:
        out = {
            "fallback_activated": 0,
            "generated": 0,
            "duplicates": 0,
        }

        for state in states:
            st = stats.get(state)
            if st is None or st.committed_skeleton is None:
                continue
            if graph.status(st.committed_skeleton) != "FAILED":
                continue

            st.skeleton_commit_failed_count += 1
            st.committed_skeleton = None
            child_stats = self.try_activate_reserved_skeleton(state, graph, stats, queue, seen_states)
            if child_stats is None:
                continue

            out["fallback_activated"] += 1
            out["generated"] += child_stats["generated"]
            out["duplicates"] += child_stats["duplicates"]

        return out

    def count_solved_children(self, action: Action, graph: ANDORGraph) -> int:
        return sum(graph.status(child) == "SOLVED" for child in action.children)

    def refresh_commitments(
        self,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
        queue: StatePriorityQueue,
        seen_states: dict[str, ProofState],
    ) -> dict[str, int]:
        out = {
            "fallback_activated": 0,
            "generated": 0,
            "duplicates": 0,
            "committed_failed": 0,
            "committed_stale": 0,
        }

        for state in graph.all_states():
            st = stats.get(state)
            if st is None or st.committed_skeleton is None:
                continue

            if graph.status(state) != "OPEN":
                continue

            skel = st.committed_skeleton
            status = graph.status(skel)

            if status == "SOLVED":
                graph.mark_solved(state)
                continue

            if status == "OPEN" and skel.children:
                if any(graph.status(child) == "FAILED" for child in skel.children):
                    graph.mark_failed(skel)
                    status = "FAILED"
                elif all(graph.status(child) == "SOLVED" for child in skel.children):
                    graph.mark_solved(skel)
                    continue

            if status == "OPEN" and skel.children:
                solved_children = self.count_solved_children(skel, graph)
                if solved_children > st.committed_skeleton_progress_last:
                    st.committed_skeleton_progress_last = solved_children
                    st.committed_skeleton_stale_rounds = 0
                    continue

                st.committed_skeleton_stale_rounds += 1
                if st.committed_skeleton_stale_rounds < self.commit_stale_rounds_before_fallback:
                    continue

                out["committed_stale"] += 1
                continue

            if status != "FAILED":
                continue

            st.skeleton_commit_failed_count += 1
            out["committed_failed"] += 1
            st.committed_skeleton = None
            child_stats = self.try_activate_reserved_skeleton(state, graph, stats, queue, seen_states)
            if child_stats is None:
                st.skeleton_exhausted = True
                continue

            out["fallback_activated"] += 1
            out["generated"] += child_stats["generated"]
            out["duplicates"] += child_stats["duplicates"]

        return out

    def record_commitment_refresh(self, metadata: dict, refresh_stats: dict[str, int]) -> None:
        metadata["skeleton_commitment"]["fallback_activated"] += refresh_stats["fallback_activated"]
        metadata["skeleton_commitment"]["committed_failed"] += refresh_stats["committed_failed"]
        metadata["skeleton_commitment"]["committed_stale"] += refresh_stats["committed_stale"]
        metadata["skeleton_pipeline"]["children_new"] += refresh_stats["generated"]
        metadata["skeleton_pipeline"]["children_duplicate"] += refresh_stats["duplicates"]

    def update_skeleton_progress(
        self,
        selected_skeletons: list[tuple[ProofState, Action, float]],
        generated_by_parent: dict[ProofState, int],
        stats: dict[ProofState, StateStats],
        active_by_parent: dict[ProofState, int] | None = None,
    ) -> None:
        touched_parents = {parent_state for parent_state, _, _ in selected_skeletons}
        for parent_state in touched_parents:
            st = stats[parent_state]
            new_children = generated_by_parent.get(parent_state, 0)
            active_children = (
                active_by_parent.get(parent_state, new_children)
                if active_by_parent is not None
                else new_children
            )
            st.last_skeleton_new_children = new_children
            if active_children == 0:
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
        if self.skeleton_commitment and st.committed_skeleton is not None:
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
