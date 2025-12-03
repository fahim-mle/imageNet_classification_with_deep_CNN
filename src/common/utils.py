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
    """
    Set the global random seed for Python, NumPy, PyTorch, and TensorFlow to make experimental runs reproducible.
    
    This function sets the Python random seed, the PYTHONHASHSEED environment variable, NumPy's RNG seed, PyTorch CPU and CUDA seeds, and TensorFlow's seed. It also configures PyTorch's cuDNN to deterministic mode and disables its benchmark mode to reduce nondeterminism across runs.
    
    Parameters:
        seed (int): Integer seed used to initialize all random number generators. Default is 42.
    """
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
    Load a YAML configuration and merge it into the project's base configuration.
    
    Performs a deep recursive merge: mappings from the provided config override or extend entries from configs/base.yaml. Nested dictionaries are merged recursively; non-dict values are replaced by the specific config.
    
    Parameters:
        config_path (str | Path): Path to the YAML config file whose values will override or extend the base config.
    
    Returns:
        dict: The resulting merged configuration dictionary.
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
        """
        Recursively merge values from `update` into `base`.
        
        Parameters:
            base (dict): Dictionary to merge into. This dictionary is modified in place.
            update (dict): Dictionary whose keys and values override or extend `base`.
        
        Returns:
            dict: The merged dictionary (the same object as `base`) where nested dictionaries are merged recursively and non-dictionary values from `update` replace those in `base`.
        """
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged_config = deep_merge(base_config, specific_config)
    return merged_config