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
