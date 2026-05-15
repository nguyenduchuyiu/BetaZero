__all__ = [
    "Lean4ServerScheduler",
]


def __getattr__(name):
    if name == "Lean4ServerScheduler":
        from .lean_verifier import Lean4ServerScheduler

        return Lean4ServerScheduler
    raise AttributeError(name)
