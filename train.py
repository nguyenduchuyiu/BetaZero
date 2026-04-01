import os
import sys

from betazero.utils.config import Config
from betazero.train import train


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    if not os.path.exists(yaml_path) and os.path.exists(os.path.join("configs", yaml_path)):
        yaml_path = os.path.join("configs", yaml_path)
    train(Config.from_yaml(yaml_path))

