from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from prover.agent.decomposer import Decomposer
from prover.agent.error_handler import ErrorHandler
from prover.agent.lean_repl import LeanRepl, LeanReplResult
from prover.agent.llm_client import LLMClient
from prover.agent.planner import Planner
from prover.agent.proof_state import NodeStatus, ProofState, RouteDecision, SubgoalNode
from prover.agent.router import Router
from prover.agent.solver import Solver
from prover.agent.sorrifier import Sorrifier
from prover.agent.verifier import ProofVerifier


@dataclass
class OrchestratorResult:
    success: bool
    message: str
    state: ProofState
    final_code: Optional[str] = None
    layers: dict[str, Any] = field(default_factory=dict)


class ProofOrchestrator:
    """
    Entry point: Layer 1 (statement → planner + skeleton sorrify) →
    Layer 2 (router per subgoal) → Layer 3 (solver + verifier + fix_proof).
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        repl: Optional[LeanRepl] = None,
        timeout: int = 300,
        max_skeleton_fixes: int = 6,
        max_solver_attempts: int = 6,
        max_decompose_depth: int = 4,
    ) -> None:
        self.llm = llm or LLMClient()
        self.repl = repl or LeanRepl(timeout=timeout)
        self.verifier = ProofVerifier(self.repl, timeout=timeout)
        self.planner = Planner(self.llm)
        self.sorrifier = Sorrifier(self.llm)
        self.errors = ErrorHandler()
        self.router = Router(self.llm)
        self.decomposer = Decomposer(self.llm)
        self.solver = Solver(self.llm)
        self.max_skeleton_fixes = max_skeleton_fixes
        self.max_solver_attempts = max_solver_attempts
        self.max_decompose_depth = max_decompose_depth

    def run(self, formal_statement: str) -> OrchestratorResult:
        state = ProofState(root_statement=formal_statement)
        layers: dict[str, Any] = {}

        r0 = self.verifier.verify(formal_statement)
        state.initial_repl_snapshot = r0.raw
        state.log("layer1_initial_repl", pass_ok=r0.pass_ok, complete=r0.complete)

        if r0.pass_ok and r0.complete:
            state.final_proof = formal_statement
            return OrchestratorResult(
                success=True,
                message="Statement already passes REPL with no sorry.",
                state=state,
                final_code=formal_statement,
                layers={"layer1": "skipped"},
            )

        skeleton = formal_statement
        if not r0.pass_ok:
            plan = self.planner.plan(formal_statement)
            state.planner_output = plan
            state.log("layer1_planner", informal_len=len(plan.informal_plan))
            skeleton = plan.skeleton_code
            skeleton, r_sk = self._sorrify_until_repl_pass(skeleton)
            layers["layer1"] = {"planner": True, "skeleton_pass_ok": r_sk.pass_ok}
        else:
            r_sk = r0
            layers["layer1"] = {"planner": False, "used_existing_skeleton": True}

        state.skeleton_repl_snapshot = r_sk.raw
        if not r_sk.pass_ok:
            return OrchestratorResult(
                success=False,
                message="Skeleton still fails REPL after sorrifier loop; escalate.",
                state=state,
                final_code=skeleton,
                layers=layers,
            )

        subgoal_texts = self._subgoal_texts(r_sk, skeleton)
        queue: deque[str] = deque()
        for text in subgoal_texts:
            node = state.new_node(text.strip() or skeleton)
            node.meta["depth"] = 0
            queue.append(node.id)

        state.log("layer2_queue_seed", n=len(queue))

        while queue:
            nid = queue.popleft()
            node = state.nodes[nid]
            depth = int(node.meta.get("depth", 0))

            decision = self.router.route(node.formal_statement)
            node.router_decision = decision
            state.log("layer2_route", node_id=nid, decision=decision.value)

            if decision == RouteDecision.DECOMPOSE and depth < self.max_decompose_depth:
                subs = self.decomposer.decompose(node.formal_statement)
                node.status = NodeStatus.DECOMPOSED
                for s in subs:
                    ch = state.new_node(s, parent_id=nid)
                    ch.meta["depth"] = depth + 1
                    queue.append(ch.id)
                continue

            ok = self._solve_node(node, state)
            if not ok:
                return OrchestratorResult(
                    success=False,
                    message=f"Subgoal failed after retries: node {node.id}",
                    state=state,
                    final_code=node.resolved_proof or skeleton,
                    layers=layers,
                )

        merged = self._merge_resolved(state, skeleton)
        state.final_proof = merged
        layers["layer3"] = {"resolved_nodes": sum(1 for n in state.nodes.values() if n.status == NodeStatus.SOLVED)}
        return OrchestratorResult(
            success=True,
            message="Pipeline finished (proof quality depends on your local LLM).",
            state=state,
            final_code=merged,
            layers=layers,
        )

    def _sorrify_until_repl_pass(self, skeleton: str) -> tuple[str, LeanReplResult]:
        code = skeleton
        last = self.verifier.verify(code)
        if last.pass_ok:
            return code, last
        for _ in range(self.max_skeleton_fixes):
            if not self.errors.can_fix(last):
                break
            code = self.sorrifier.fix_proof(code, self.errors.summarize(last))
            last = self.verifier.verify(code)
            if last.pass_ok:
                return code, last
        return code, last

    @staticmethod
    def _subgoal_texts(r: LeanReplResult, skeleton: str) -> list[str]:
        goals = [str(s.get("goal") or "").strip() for s in r.sorries]
        goals = [g for g in goals if g]
        if goals:
            return goals
        return [skeleton]

    def _solve_node(self, node: SubgoalNode, state: ProofState) -> bool:
        node.status = NodeStatus.IN_PROGRESS
        proof = self.solver.solve(node.formal_statement)
        last: Optional[LeanReplResult] = None
        for attempt in range(1, self.max_solver_attempts + 1):
            last = self.verifier.verify(proof)
            state.log(
                "layer3_verify",
                node_id=node.id,
                attempt=attempt,
                pass_ok=last.pass_ok,
                complete=last.complete,
            )
            if last.pass_ok and last.complete:
                node.resolved_proof = proof
                node.status = NodeStatus.SOLVED
                return True
            if not self.errors.can_fix(last):
                node.status = NodeStatus.ESCALATED
                node.last_error_summary = self.errors.summarize(last)
                return False
            proof = self.sorrifier.fix_proof(proof, self.errors.summarize(last))
            node.retry_count = attempt
        node.status = NodeStatus.FAILED
        if last:
            node.last_error_summary = self.errors.summarize(last)
        return False

    @staticmethod
    def _merge_resolved(state: ProofState, skeleton: str) -> str:
        parts = [n.resolved_proof for n in state.nodes.values() if n.resolved_proof]
        if not parts:
            return skeleton
        return "\n\n-- --- merged subproofs ---\n\n".join(parts)


if __name__ == "__main__":
    import os

    # Chạy nhanh không cần server: chỉ khi chưa set LOCAL_LLM_BASE_URL thì dùng stub (LOCAL_LLM_MOCK).
    if not (os.environ.get("LOCAL_LLM_BASE_URL") or "").strip():
        os.environ.setdefault("LOCAL_LLM_MOCK", "1")

    demo = "import Mathlib\n\ntheorem demo_orchestrator : True := by\n  sorry\n"
    orch = ProofOrchestrator(timeout=120)
    out = orch.run(demo)
    print("success:", out.success)
    print("message:", out.message)
    print("layers:", out.layers)
