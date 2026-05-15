__all__ = [
    "ANDORGraph",
    "BatchExecutor",
    "BestFirstRollout",
    "DependencyRewardAssigner",
    "FailureHandler",
    "GRPOTrainer",
    "LevelwiseRollout",
    "RewardCalculator",
    "RolloutBudget",
    "SamplePolicy",
    "Sorrifier",
]


def __getattr__(name):
    if name == "ANDORGraph":
        from .graph import ANDORGraph

        return ANDORGraph
    if name in {"DependencyRewardAssigner", "RewardCalculator"}:
        from .reward import DependencyRewardAssigner, RewardCalculator

        return {"DependencyRewardAssigner": DependencyRewardAssigner, "RewardCalculator": RewardCalculator}[name]
    if name in {"BatchExecutor", "BestFirstRollout", "FailureHandler", "LevelwiseRollout", "RolloutBudget", "SamplePolicy"}:
        from .rollout import BatchExecutor, BestFirstRollout, FailureHandler, LevelwiseRollout, RolloutBudget, SamplePolicy

        return {
            "BatchExecutor": BatchExecutor,
            "BestFirstRollout": BestFirstRollout,
            "FailureHandler": FailureHandler,
            "LevelwiseRollout": LevelwiseRollout,
            "RolloutBudget": RolloutBudget,
            "SamplePolicy": SamplePolicy,
        }[name]
    if name == "Sorrifier":
        from .sorrifier import Sorrifier

        return Sorrifier
    if name == "GRPOTrainer":
        from .trainer import GRPOTrainer

        return GRPOTrainer
    raise AttributeError(name)
