import logging
import random
import numpy as np
import os
import torch
import tensorflow as tf
from .paths import LOGS_DIR

def setup_logging(log_file="app.log"):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ImageNet_CNN")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)

import yaml
from pathlib import Path

def load_config(config_path):
    """
    Loads a config file and merges it with base.yaml.

    Args:
        config_path (str): Path to the specific config file.

    Returns:
        dict: Merged configuration.
    """
    # Load base config
    base_config_path = Path("configs/base.yaml")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load specific config
    specific_config_path = Path(config_path)
    with open(specific_config_path, 'r') as f:
        specific_config = yaml.safe_load(f)

    # Merge configs (simple recursive merge or just top-level update?)
    # Instructions say "Values must merge with base.yaml".
    # Let's do a deep merge for 'dataset', 'training', 'model', etc.

    def deep_merge(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged_config = deep_merge(base_config, specific_config)
    return merged_config
