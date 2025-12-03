import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.common.utils import load_config
from src.tensorflow.train import train_model


def main():
    cfg = load_config("configs/tensorflow.yaml")
    train_model(cfg)


if __name__ == "__main__":
    main()
    main()
