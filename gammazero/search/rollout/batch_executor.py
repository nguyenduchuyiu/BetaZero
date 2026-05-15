from __future__ import annotations

import concurrent.futures
import threading
from typing import TYPE_CHECKING

from gammazero.core import ProofState, Action
from gammazero.policy.output_parser import get_lean_code
from gammazero.policy.prompt import build_prompt
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward import RewardCalculator
from gammazero.utils.lean_cmd import build_theorem

from .execution_result import LeanExecutionResult
from .failure_handler import FailedActionPatch, FailureHandler
from .utils import format_lean_feedback

if TYPE_CHECKING:
    from gammazero.env.lean_env import LeanEnv


class RolloutBudget:
    __slots__ = ("max_nodes", "used", "_lock")

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.used = 0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.used = 0

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
                sc = build_theorem(state, action_code)
            except Exception:
                sc = ""
            return LeanExecutionResult.from_transport_error(f"{type(e).__name__}: {e}", sc)

    def execute(
        self,
        graph: ANDORGraph,
        states: list[ProofState],
        action_batches: list[list[dict]],
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
        patch_futures: list[
            tuple[int, int, str, dict, concurrent.futures.Future[FailedActionPatch]]
        ] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for i, (state, actions) in enumerate(zip(states, action_batches)):
                for j, action_dict in enumerate(actions):
                    if not budget.try_consume():
                        break
                    raw_output = action_dict["text"]
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
                full_code = build_theorem(state, lean_code)
                # For a complete tactic, code passed Lean with 0 sorries → r_env = 1.0
                r_env = 1.0

                if state_vr.get("complete"):
                    if action_type not in ("tactic", "skeleton"):
                        raise ValueError(f"Invalid action type: {action_type}")
                    act = Action(
                        action_type=action_type,
                        content=raw_output,
                        extracted_code=lean_code,
                        children=(),
                        prompt=prompt,
                    )
                    graph.expand(
                        state,
                        act,
                        r_env=r_env,
                        tactic_status="SOLVED" if action_type == "tactic" else None,
                    )
                    if action_type == "skeleton":
                        graph.set_skeleton_override(act, True)
                elif action_type == "tactic":
                    fut = pool.submit(
                        self.failure.compute_failed_action_patch,
                        state,
                        action_type,
                        raw_output,
                        lean_code,
                        state_code,
                        state_vr,
                        prompt,
                    )
                    patch_futures.append((i, j, lean_code, state_vr, fut))
                    
                elif action_type == "skeleton":
                    if state_vr.get("pass"):
                        # Calculate r_env even for passing skeletons to catch semantic/AST issues
                        r_env_score = self.reward.r_env(full_code, full_code, state_vr)
                        graph.expand(
                            state,
                            Action(
                                action_type="skeleton",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=tuple(subgoals),
                                prompt=prompt,
                            ),
                            r_env=r_env_score,
                        )
                    else:
                        fut = pool.submit(
                            self.failure.compute_failed_action_patch,
                            state,
                            action_type,
                            raw_output,
                            lean_code,
                            state_code,
                            state_vr,
                            prompt,
                        )
                        patch_futures.append((i, j, lean_code, state_vr, fut))
                else:
                    raise ValueError(f"Invalid action type: {action_type}")

            for i, j, lean_code, state_vr, future in patch_futures:
                patch = future.result()
                sorr_body = self.failure.apply_failed_action_patch(graph, patch)
                feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), sorr_body)

        return feedbacks
