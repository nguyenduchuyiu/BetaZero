"""Rollout pipeline for sampling, execution, repair, and graph expansion."""

__all__ = [
    "BatchExecutor",
    "DependencyRewardAssigner",
    "FailureHandler",
    "LeanExecutionResult",
    "BestFirstRollout",
    "LevelwiseRollout",
    "RolloutBudget",
    "SamplePolicy",
]


def __getattr__(name):
    if name == "LeanExecutionResult":
        from .execution_result import LeanExecutionResult

        return LeanExecutionResult
    if name in {"BatchExecutor", "RolloutBudget"}:
        from .batch_executor import BatchExecutor, RolloutBudget

        return {"BatchExecutor": BatchExecutor, "RolloutBudget": RolloutBudget}[name]
    if name == "FailureHandler":
        from .failure_handler import FailureHandler

        return FailureHandler
    if name in {"BestFirstRollout", "SamplePolicy"}:
        from .best_first_rollout import BestFirstRollout, SamplePolicy

        return {"BestFirstRollout": BestFirstRollout, "SamplePolicy": SamplePolicy}[name]
    if name == "LevelwiseRollout":
        from .levelwise_rollout import LevelwiseRollout

        return LevelwiseRollout
    if name == "DependencyRewardAssigner":
        from gammazero.search.reward import DependencyRewardAssigner

        return DependencyRewardAssigner
    raise AttributeError(name)
