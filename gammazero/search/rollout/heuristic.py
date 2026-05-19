from __future__ import annotations

from gammazero.core import Action, ProofState
from gammazero.search.graph import ANDORGraph

from .search_stats import StateStats


class SearchScorer:
    def score_state(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        raise NotImplementedError

    def score_skeleton(
        self,
        action: Action,
        parent_state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        raise NotImplementedError


class SimpleHeuristicScorer(SearchScorer):
    def __init__(
        self,
        *,
        incoming_skeleton_weight: float = 2.0,
        best_tactic_weight: float = 1.5,
        depth_penalty: float = 0.15,
        tactic_retry_penalty: float = 0.10,
        skeleton_retry_penalty: float = 0.12,
        bad_skeleton_round_penalty: float = 0.8,
        skeleton_r_env_weight: float = 2.0,
        skeleton_parent_score_weight: float = 0.6,
        skeleton_depth_penalty: float = 0.25,
        skeleton_sorrified_penalty: float = 0.25,
        committed_child_bonus: float = 8.0,
        last_child_committed_bonus: float = 20.0,
    ):
        self.incoming_skeleton_weight = incoming_skeleton_weight
        self.best_tactic_weight = best_tactic_weight
        self.depth_penalty = depth_penalty
        self.tactic_retry_penalty = tactic_retry_penalty
        self.skeleton_retry_penalty = skeleton_retry_penalty
        self.bad_skeleton_round_penalty = bad_skeleton_round_penalty
        self.skeleton_r_env_weight = skeleton_r_env_weight
        self.skeleton_parent_score_weight = skeleton_parent_score_weight
        self.skeleton_depth_penalty = skeleton_depth_penalty
        self.skeleton_sorrified_penalty = skeleton_sorrified_penalty
        self.committed_child_bonus = committed_child_bonus
        self.last_child_committed_bonus = last_child_committed_bonus

    def committed_skeleton_progress_bonus(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        bonus = 0.0
        for parent_state, skeleton in getattr(stats[state], "parent_skeletons", []):
            parent_stats = stats.get(parent_state)
            if parent_stats is None or parent_stats.committed_skeleton != skeleton:
                continue

            children = skeleton.children
            if not children:
                continue
            statuses = [graph.status(child) for child in children]
            if any(status == "FAILED" for status in statuses):
                continue

            bonus += self.committed_child_bonus

            open_children = [
                child
                for child, status in zip(children, statuses, strict=False)
                if status == "OPEN"
            ]
            if len(open_children) == 1 and open_children[0] == state:
                bonus += self.last_child_committed_bonus

        return bonus

    def score_state(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        st = stats[state]

        score = 0.0
        score += self.incoming_skeleton_weight * getattr(st, "incoming_skeleton_score", 0.0)
        score += self.best_tactic_weight * getattr(st, "best_tactic_r_env", 0.0)
        score += self.committed_skeleton_progress_bonus(state, graph, stats)
        score -= self.depth_penalty * st.depth
        score -= self.tactic_retry_penalty * st.tactic_tries
        score -= self.skeleton_retry_penalty * st.skeleton_tries
        score -= self.bad_skeleton_round_penalty * getattr(st, "bad_skeleton_rounds", 0)
        return score

    def score_skeleton(
        self,
        action: Action,
        parent_state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        parent_stats = stats[parent_state]

        n_children = len(action.children)
        r_env = graph.get_r_env(action)

        score = 0.0
        score += self.skeleton_r_env_weight * r_env
        score += self.skeleton_parent_score_weight * getattr(parent_stats, "last_score", 0.0)
        score += self.child_count_score(n_children)
        score -= self.skeleton_depth_penalty * parent_stats.depth

        if getattr(action, "was_sorrified", False):
            score -= self.skeleton_sorrified_penalty

        return score

    def child_count_score(self, n: int) -> float:
        if n <= 0:
            return -2.0
        return 1.0 - 0.25 * max(0, n - 2)
