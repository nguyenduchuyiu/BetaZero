"""Rollout pipeline for sampling, execution, repair, and graph expansion."""

__all__ = [
    "BatchExecutor",
    "DependencyRewardAssigner",
    "FailureHandler",
    "LeanExecutionResult",
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
    if name in {"LevelwiseRollout", "SamplePolicy"}:
        from .levelwise_rollout import LevelwiseRollout, SamplePolicy

        return {"LevelwiseRollout": LevelwiseRollout, "SamplePolicy": SamplePolicy}[name]
    if name == "DependencyRewardAssigner":
        from betazero.search.reward import DependencyRewardAssigner

        return DependencyRewardAssigner
    raise AttributeError(name)
