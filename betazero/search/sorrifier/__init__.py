"""Sorrifier for patching failed tactics/skeletons."""

__all__ = ["Sorrifier"]


def __getattr__(name):
    if name == "Sorrifier":
        from .sorrifier import Sorrifier

        return Sorrifier
    raise AttributeError(name)
