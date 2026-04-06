from __future__ import annotations

import concurrent.futures
import threading

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.policy.prompt import build_prompt
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator

from .execution_result import LeanExecutionResult
from .failure_handler import FailureHandler
from .utils import format_lean_feedback


class RolloutBudget:
    __slots__ = ("max_nodes", "used", "_lock")

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        with self._lock:
            if self.used >= self.max_nodes:
                return False
            self.used += 1
            return True


class BatchExecutor:
    """Parallel Lean execute + expand graph; tactic feedbacks align with action_batches[i][j]."""

    def __init__(
        self,
        lean: LeanEnv,
        failure_handler: FailureHandler,
        reward: RewardCalculator,
        max_workers: int | None = None,
    ):
        self.lean = lean
        self.failure = failure_handler
        self.reward = reward
        # Get max workers from the executor to synchronize, avoid context switching.
        ex = getattr(lean.scheduler, "executor", None)
        self._max_workers = max_workers if max_workers is not None else (
            getattr(ex, "_max_workers", 4) if ex is not None else 4
        )

    @staticmethod
    def safe_execute(lean: LeanEnv, state: ProofState, action_code: str) -> LeanExecutionResult:
        """Run Lean; never raises — transport/executor errors become `system_errors` on the result."""
        try:
            sc, vr, sg = lean.execute(state, action_code)
            return LeanExecutionResult.ok(sc, vr, sg)
        except Exception as e:
            try:
                sc = lean._build_cmd(state, action_code)
            except Exception:
                sc = ""
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}", sc)

    def execute(
        self,
        graph: ANDORGraph,
        states: list[ProofState],
        action_batches: list[list[str]],
        action_type: str,
        budget: RolloutBudget,
        prompts: list[str] | None = None,
    ) -> list[list[tuple[str, str, str] | None]]:
        if prompts is None:
            prompts = [build_prompt(s, action_type) for s in states]

        tasks: list[tuple[int, int, ProofState, str, str, concurrent.futures.Future]] = []
        feedbacks: list[list[tuple[str, str, str] | None]] = [
            [None] * len(actions) for actions in action_batches
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for i, (state, actions) in enumerate(zip(states, action_batches)):
                for j, raw_output in enumerate(actions):
                    if not budget.try_consume():
                        break
                    lean_code = get_lean_code(raw_output)
                    if not lean_code:
                        self.failure.handle_system_execute_failure(
                            graph,
                            state,
                            action_type,
                            raw_output,
                            LeanExecutionResult.from_transport_error("empty_lean_code"),
                            prompts[i],
                        )
                        continue
                    fut = pool.submit(BatchExecutor.safe_execute, self.lean, state, lean_code)
                    tasks.append((i, j, state, raw_output, lean_code, fut))
                if budget.used >= budget.max_nodes:
                    break

            for i, j, state, raw_output, lean_code, future in tasks:
                res: LeanExecutionResult = future.result()
                prompt = prompts[i]
                if res.has_system_failure:
                    self.failure.handle_system_execute_failure(
                        graph, state, action_type, raw_output, res, prompt
                    )
                    continue
                state_code, state_vr, subgoals = res.state_code, res.verify, list(res.subgoals)
                r_env = self.reward.r_env(state_code, state_code, state_vr)
                if state_vr.get("complete"):
                    if action_type not in ("tactic", "skeleton"):
                        raise ValueError(f"Invalid action type: {action_type}")
                    act = Action(action_type, raw_output, (), prompt=prompt)
                    graph.expand(
                        state,
                        act,
                        r_env=r_env,
                        tactic_status="SOLVED" if action_type == "tactic" else None,
                    )
                    if action_type == "skeleton":
                        graph.set_skeleton_override(act, True)
                elif action_type == "tactic":
                    sorr_body = self.failure.handle_failed_tactic(
                        graph, state, raw_output, state_code, state_vr, prompt
                    )
                    feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), sorr_body)
                elif action_type == "skeleton":
                    if state_vr.get("pass"):
                        graph.expand(
                            state,
                            Action("skeleton", raw_output, tuple(subgoals), prompt=prompt),
                            r_env=r_env,
                        )
                    else:
                        self.failure.handle_failed_skeleton(
                            graph, state, raw_output, state_code, state_vr, prompt
                        )
                else:
                    raise ValueError(f"Invalid action type: {action_type}")

        return feedbacks
