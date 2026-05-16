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
        final_child_bonus: float = 6.0,
        solved_ratio_bonus: float = 2.0,
        parent_depth_bonus: float = 1.0,
        depth_penalty: float = 0.10,
        tactic_retry_penalty: float = 0.10,
        skeleton_retry_penalty: float = 0.12,
        bad_skeleton_round_penalty: float = 0.8,
        skeleton_r_env_weight: float = 2.0,
        skeleton_parent_score_weight: float = 0.6,
        skeleton_depth_penalty: float = 0.25,
        skeleton_sorrified_penalty: float = 0.25,
    ):
        self.incoming_skeleton_weight = incoming_skeleton_weight
        self.best_tactic_weight = best_tactic_weight
        self.final_child_bonus = final_child_bonus
        self.solved_ratio_bonus = solved_ratio_bonus
        self.parent_depth_bonus = parent_depth_bonus
        self.depth_penalty = depth_penalty
        self.tactic_retry_penalty = tactic_retry_penalty
        self.skeleton_retry_penalty = skeleton_retry_penalty
        self.bad_skeleton_round_penalty = bad_skeleton_round_penalty
        self.skeleton_r_env_weight = skeleton_r_env_weight
        self.skeleton_parent_score_weight = skeleton_parent_score_weight
        self.skeleton_depth_penalty = skeleton_depth_penalty
        self.skeleton_sorrified_penalty = skeleton_sorrified_penalty

    def parent_completion_bonus(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        bonus = 0.0
        for parent_state, skeleton in getattr(stats[state], "parent_skeletons", []):
            children = skeleton.children
            if not children:
                continue
            if any(graph.status(child) == "FAILED" for child in children):
                continue

            total = len(children)
            solved = sum(graph.status(child) == "SOLVED" for child in children)
            open_count = sum(graph.status(child) == "OPEN" for child in children)
            solved_ratio = solved / total

            if open_count == 1 and graph.status(state) == "OPEN":
                bonus += self.final_child_bonus
            bonus += self.solved_ratio_bonus * solved_ratio

            parent_stats = stats.get(parent_state)
            parent_depth = parent_stats.depth if parent_stats is not None else graph.get_depth(parent_state)
            bonus += self.parent_depth_bonus / (1.0 + max(parent_depth, 0))
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
        score += self.parent_completion_bonus(state, graph, stats)
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
        if n == 0:
            return -2.0
        if n == 1:
            return 0.2
        if n in (2, 3):
            return 1.0
        if n == 4:
            return 0.5
        return -0.3 * (n - 4)
