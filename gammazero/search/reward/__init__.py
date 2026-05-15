"""Reward computation for local execution quality and dependency structure."""

__all__ = ["DependencyRewardAssigner", "RewardCalculator"]


def __getattr__(name):
    if name == "RewardCalculator":
        from .calculator import RewardCalculator

        return RewardCalculator
    if name == "DependencyRewardAssigner":
        from .reward_assigner import DependencyRewardAssigner

        return DependencyRewardAssigner
    raise AttributeError(name)
