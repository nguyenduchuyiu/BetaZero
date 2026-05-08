================================================================================
### betazero/search/rollout/batch_executor.py ###
================================================================================
from __future__ import annotations

import concurrent.futures
import threading

from betazero.core import ProofState, Action
from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.policy.prompt import build_prompt
from betazero.search.graph import ANDORGraph
from betazero.search.reward import RewardCalculator
from betazero.utils.lean_cmd import build_theorem

from .execution_result import LeanExecutionResult
from .failure_handler import FailureHandler
from .utils import format_lean_feedback


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
                r_env = self.reward.r_env(state_code, state_code, state_vr)
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
                    sorr_body = self.failure.handle_failed_tactic(
                        graph, state, raw_output, lean_code, state_code, state_vr, prompt
                    )
                    feedbacks[i][j] = (lean_code, format_lean_feedback(state_vr), sorr_body)
                elif action_type == "skeleton":
                    if state_vr.get("pass"):
                        graph.expand(
                            state,
                            Action(
                                action_type="skeleton",
                                content=raw_output,
                                extracted_code=lean_code,
                                children=tuple(subgoals),
                                prompt=prompt,
                            ),
                            r_env=r_env,
                        )
                    else:
                        self.failure.handle_failed_skeleton(
                            graph,
                            state,
                            raw_output,
                            lean_code,
                            state_code,
                            state_vr,
                            prompt,
                        )
                else:
                    raise ValueError(f"Invalid action type: {action_type}")

        return feedbacks


================================================================================
### betazero/search/reward/reward_assigner.py ###
================================================================================
"""Assigns structural dependencies reward (r_dep) to skeleton actions."""

from betazero.env.lean_env import LeanEnv
from betazero.policy.output_parser import get_lean_code
from betazero.search.graph import ANDORGraph
from betazero.search.sorrifier.stitcher import ProofStitcher
from betazero.utils.lean_cmd import build_theorem
from .calculator import RewardCalculator


class DependencyRewardAssigner:
    """Orchestrates stitching and kernel analysis to assign dependency rewards."""

    def __init__(self, lean: LeanEnv, reward: RewardCalculator):
        self.lean = lean
        self.reward = reward

    def assign(self, graph: ANDORGraph) -> None:
        """Bottom-up dependency reward assignment using Expr Trees."""
        for action, parent_state in graph.parent_items():
            if action.action_type != "skeleton":
                continue
            
            # 1. Collect child proofs
            child_proofs = [graph.extract_proof_code(child) for child in action.children]
            
            # 2. Stitch code
            stitched_code = ProofStitcher.stitch(action.extracted_code, child_proofs)
            full_compilable_code = build_theorem(parent_state, stitched_code)
            
            # 3. Analyze through Kernel Expr Tree
            dep_analysis = self.lean.analyze_dependencies(full_compilable_code)
            
            # 4. Map outputs to Calculator format and assign
            mapped_analysis = {
                "core": dep_analysis.get("core_solved", []) + dep_analysis.get("core_failed", []),
                "benign": dep_analysis.get("benign", []),
                "malignant": dep_analysis.get("malignant", [])
            }
            
            r_dep_score = self.reward.r_dep(mapped_analysis)
            
            # Fatal penalty for missing core subgoals
            if dep_analysis.get("core_failed"):
                r_dep_score = -1.0 
                
            graph.set_r_dep(action, r_dep_score)

================================================================================
### betazero/search/graph/and_or_graph.py ###
================================================================================
from __future__ import annotations

import threading
from typing import Any, Literal

from betazero.core.nodes import Action, NodeStatus, ProofState
from betazero.policy.output_parser import get_lean_code
from betazero.search.sorrifier.stitcher import ProofStitcher


class ANDORGraph:
    """Thread-safe AND/OR proof graph with solved-state checks and return backup."""

    def __init__(self, root: ProofState):
        self._lock = threading.RLock()
        self._actions: dict[ProofState, list[Action]] = {root: []}
        self._parent: dict[Action, ProofState] = {}
        self._r_env: dict[Action, float] = {}
        self._r_dep: dict[Action, float] = {}
        self._tactic_status: dict[Action, Literal["SOLVED", "FAILED"]] = {}
        self._depth: dict[ProofState, int] = {root: 0}
        self._solved_cache: dict[Any, bool] = {}
        self._skeleton_override: dict[Action, bool] = {} 

    def expand(
        self,
        state: ProofState,
        action: Action,
        r_env: float = 0.0,
        r_dep: float = 0.0,
        tactic_status: Literal["SOLVED", "FAILED"] | None = None,
    ) -> None:
        with self._lock:
            if action in self._parent:
                return
            self._solved_cache.clear()
            self._actions.setdefault(state, []).append(action)
            self._parent[action] = state
            self._r_env[action] = r_env
            self._r_dep[action] = r_dep
            if tactic_status is not None and action.action_type == "tactic":
                self._tactic_status[action] = tactic_status
            for child in action.children:
                self._actions.setdefault(child, [])
                if child not in self._depth:
                    self._depth[child] = self._depth[state] + 1

    def _node_solved(
        self, node: ProofState | Action, visiting: set, memo: dict[Any, bool]
    ) -> bool:
        if node in visiting:
            return False
        if node in memo:
            return memo[node]
        visiting.add(node)
        try:
            if isinstance(node, ProofState):
                res = any(self._node_solved(a, visiting, memo) for a in self._actions.get(node, []))
            elif node.action_type == "tactic":
                res = self._tactic_status.get(node) == "SOLVED"
            else:
                if node in self._skeleton_override:
                    res = self._skeleton_override[node]
                else:
                    res = bool(node.children) and all(
                        self._node_solved(c, visiting, memo) for c in node.children
                    )
            memo[node] = res
            return res
        finally:
            visiting.remove(node)

    def is_solved(self, node: ProofState | Action, visiting: set | None = None) -> bool:
        with self._lock:
            if visiting is None:
                visiting = set()
            return self._node_solved(node, visiting, self._solved_cache)

    def status(self, node: ProofState | Action) -> NodeStatus:
        with self._lock:
            if isinstance(node, ProofState):
                return "SOLVED" if self.is_solved(node) else "OPEN"
            if node.action_type == "tactic":
                t = self._tactic_status.get(node)
                if t == "SOLVED":
                    return "SOLVED"
                if t == "FAILED":
                    return "FAILED"
                return "OPEN"
            if self.is_solved(node):
                return "SOLVED"
            if not node.children:
                return "FAILED"
            return "OPEN"

    def unsolved_states(self) -> list[ProofState]:
        with self._lock:
            keys = list(self._actions.keys())
        return [s for s in keys if not self.is_solved(s)]

    def get_actions(self, state: ProofState) -> list[Action]:
        with self._lock:
            return list(self._actions.get(state, []))

    def get_r_env(self, action: Action) -> float:
        with self._lock:
            return self._r_env.get(action, 0.0)

    def get_parent(self, action: Action, default: ProofState | None = None) -> ProofState | None:
        with self._lock:
            return self._parent.get(action, default)

    def parent_items(self) -> list[tuple[Action, ProofState]]:
        with self._lock:
            return list(self._parent.items())

    def set_r_dep(self, action: Action, r_dep: float) -> None:
        with self._lock:
            self._r_dep[action] = r_dep

    def set_skeleton_override(self, action: Action, is_solved: bool):
        with self._lock:
            self._skeleton_override[action] = is_solved
            self._solved_cache.clear() # Nhớ xóa cache để graph tính lại từ đầu

    def get_depth(self, state: ProofState) -> int:
        with self._lock:
            return self._depth.get(state, -1)

    def backup(self, gamma: float = 1.0, W_solve: float = 1.0) -> dict[Action, float]:
        with self._lock:
            q_cache: dict[Action, float] = {}
            v_cache: dict[ProofState, float] = {}
            visiting_v: set[ProofState] = set()
            solve_memo: dict[Any, bool] = {}

            def V(state: ProofState) -> float:
                if state in v_cache:
                    return v_cache[state]
                if state in visiting_v:
                    return 0.0
                visiting_v.add(state)
                val = max((Q(a) for a in self._actions.get(state, [])), default=0.0)
                visiting_v.remove(state)
                v_cache[state] = val
                return val

            def Q(action: Action) -> float:
                if action in q_cache:
                    return q_cache[action]
                r_e = self._r_env.get(action, 0.0)
                solved = self._node_solved(action, set(), solve_memo)
                if action.action_type == "tactic":
                    val = r_e + W_solve * float(solved)
                else:
                    r_d = self._r_dep.get(action, 0.0)
                    future = gamma * min((V(c) for c in action.children), default=0.0)
                    val = r_e + float(solved) * (r_d + future)
                q_cache[action] = val
                return val

            for action in self._parent:
                Q(action)
            for state in self._actions:
                V(state)
            return q_cache

    def get_successful_action(self, state: ProofState) -> Action | None:
        """Retrieve the action that successfully solved this state."""
        with self._lock:
            for action in self.get_actions(state):
                if self.status(action) == "SOLVED":
                    return action
        return None

    def extract_proof_code(self, state: ProofState) -> str | None:
        """Recursively extract and stitch the successful proof code for a state."""
        
        action = self.get_successful_action(state)
        if not action:
            return None

        if action.action_type == "tactic":
            return action.extracted_code
            
        # Skeleton: recurse down to children
        child_proofs = [self.extract_proof_code(child) for child in action.children]
        return ProofStitcher.stitch(action.extracted_code, child_proofs)
