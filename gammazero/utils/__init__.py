__all__ = [
    "Config",
    "TheoremDataset",
    "setup_logger",
]


def __getattr__(name):
    if name == "Config":
        from .config import Config

        return Config
    if name == "TheoremDataset":
        from .dataloader import TheoremDataset

        return TheoremDataset
    if name == "setup_logger":
        from .logger import setup as setup_logger

        return setup_logger
    raise AttributeError(name)
