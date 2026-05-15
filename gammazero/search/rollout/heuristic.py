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
    def score_state(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        st = stats[state]

        score = 0.0
        score += 2.0 * getattr(st, "incoming_skeleton_score", 0.0)
        score += 1.5 * getattr(st, "best_tactic_r_env", 0.0)
        score -= 0.35 * st.depth
        score -= 0.08 * st.tactic_tries
        score -= 0.12 * st.skeleton_tries
        score -= 0.7 * getattr(st, "bad_skeleton_rounds", 0)
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
        score += 2.0 * r_env
        score += 0.6 * getattr(parent_stats, "last_score", 0.0)
        score += self.child_count_score(n_children)
        score -= 0.25 * parent_stats.depth

        if getattr(action, "was_sorrified", False):
            score -= 0.25

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
