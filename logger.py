import logging
import os
from torch.utils.tensorboard import SummaryWriter


def setup(log_dir: str) -> tuple[logging.Logger, SummaryWriter]:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("betazero")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)

    writer = SummaryWriter(log_dir=log_dir)
    return logger, writer
