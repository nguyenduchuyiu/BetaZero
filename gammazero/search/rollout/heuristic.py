from __future__ import annotations

from typing import Protocol

from gammazero.core import Action, ProofState
from gammazero.search.graph import ANDORGraph

from .search_stats import StateStats


class HeuristicScorer(Protocol):
    def score_state(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float: ...

    def score_skeleton(
        self,
        action: Action,
        parent_state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float: ...


class DefaultScorer:
    def score_state(
        self,
        state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        st = stats[state]
        return -st.depth - 0.01 * st.tactic_tries - 0.01 * st.skeleton_tries

    def score_skeleton(
        self,
        action: Action,
        parent_state: ProofState,
        graph: ANDORGraph,
        stats: dict[ProofState, StateStats],
    ) -> float:
        return -abs(len(action.children) - 2)
