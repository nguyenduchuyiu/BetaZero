"""Policy optimization routines over collected search samples."""

__all__ = ["GRPOTrainer"]


def __getattr__(name):
    if name == "GRPOTrainer":
        from .grpo_trainer import GRPOTrainer

        return GRPOTrainer
    raise AttributeError(name)
