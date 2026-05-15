from __future__ import annotations

import heapq
import itertools

from gammazero.core import ProofState


class StatePriorityQueue:
    def __init__(self):
        self.heap: list[tuple[float, int, ProofState]] = []
        self.counter = itertools.count()

    def push(self, state: ProofState, score: float) -> None:
        heapq.heappush(self.heap, (-score, next(self.counter), state))

    def pop(self) -> tuple[ProofState | None, float | None]:
        while self.heap:
            neg_score, _, state = heapq.heappop(self.heap)
            return state, -neg_score
        return None, None

    def __len__(self) -> int:
        return len(self.heap)

    def items(self) -> list[tuple[float, ProofState]]:
        return [(-neg_score, state) for neg_score, _, state in self.heap]

    def rebuild(self, scored_states: list[tuple[ProofState, float]]) -> None:
        self.heap.clear()
        for state, score in scored_states:
            self.push(state, score)
