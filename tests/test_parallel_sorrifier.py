import pytest

from gammazero.core.nodes import ProofState
from gammazero.env.lean_env import LeanEnv
from gammazero.env.lean_verifier import Lean4ServerScheduler
from gammazero.search.graph import ANDORGraph
from gammazero.search.reward.calculator import RewardCalculator
from gammazero.search.rollout.batch_executor import BatchExecutor, RolloutBudget
from gammazero.search.rollout.failure_handler import FailureHandler
from gammazero.search.sorrifier.sorrifier import Sorrifier


def _raw_theorem(body: str) -> str:
    return f"```lean4\ntheorem my_theorem : True := by\n{body}\n```"


@pytest.fixture()
def real_rollout_parts():
    scheduler = Lean4ServerScheduler(
        max_concurrent_requests=2,
        timeout=60,
        name="test_parallel_sorrifier",
    )
    try:
        lean = LeanEnv(scheduler)
        reward = RewardCalculator()
        sorrifier = Sorrifier(scheduler, max_cycles=8)
        failure = FailureHandler(lean, sorrifier, reward)
        executor = BatchExecutor(lean, failure, reward, max_workers=2)
        yield lean, executor
    finally:
        scheduler.close()


def test_real_batch_executor_sorrifies_failed_tactics_and_updates_graph(real_rollout_parts):
    _, executor = real_rollout_parts
    root = ProofState(context="", goal="True", header="")
    graph = ANDORGraph(root)

    feedbacks = executor.execute(
        graph,
        [root],
        [
            [
                {"text": _raw_theorem("  exact missing_one")},
                {"text": _raw_theorem("  exact missing_two")},
            ]
        ],
        "tactic",
        RolloutBudget(10),
        prompts=["prompt"],
    )

    actions = graph.get_actions(root)
    assert len(actions) == 2
    assert all(graph.status(action) == "FAILED" for action in actions)
    assert sorted(action.extracted_code.strip() for action in actions) == [
        "exact missing_one",
        "exact missing_two",
    ]
    assert all(cell is not None for cell in feedbacks[0])
    assert all("sorry" in cell[2] for cell in feedbacks[0])


def test_real_batch_executor_scores_failed_skeleton_without_synthetic_branch(real_rollout_parts):
    _, executor = real_rollout_parts
    root = ProofState(context="", goal="True", header="")
    graph = ANDORGraph(root)

    feedbacks = executor.execute(
        graph,
        [root],
        [
            [
                {
                    "text": _raw_theorem(
                        "  have h : True := by\n"
                        "    exact missing_skeleton_proof\n"
                        "  exact h"
                    )
                }
            ]
        ],
        "skeleton",
        RolloutBudget(10),
        prompts=["prompt"],
    )

    actions = graph.get_actions(root)

    assert len(actions) == 1
    assert actions[0].action_type == "skeleton"
    assert actions[0].children == ()
    assert graph.status(actions[0]) == "FAILED"
    assert graph.get_r_env(actions[0]) >= 0.0
    assert feedbacks[0][0] is not None
