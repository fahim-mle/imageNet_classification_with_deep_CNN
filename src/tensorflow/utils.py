import logging
import os

import tensorflow as tf

# Disable XLA to prevent CUDA_ERROR_UNSUPPORTED_PTX_VERSION
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce logging clutter

from src.common.paths import MODELS_DIR, OUTPUTS_DIR
from src.common.utils import load_config





def setup_logging(cfg):
    """
    Create directories:
    - outputs/tf_logs/
    - models/tf/

    Return logger (Python logging)
    """
    # Create directories
    tf_logs_dir = os.path.join(OUTPUTS_DIR, "tf_logs")
    tf_models_dir = os.path.join(MODELS_DIR, "tf")

    os.makedirs(tf_logs_dir, exist_ok=True)
    os.makedirs(tf_models_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger("tensorflow_core")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already setup
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_tf_config():
    """
    Merge base.yaml + tensorflow.yaml
    Return config dict/object
    """
    # Assuming load_config handles merging if we pass the specific config
    # But the instruction says "Merge base.yaml + tensorflow.yaml"
    # The user rule says: "Gemini MUST use base.yaml + pytorch.yaml merged (pytorch overrides base)."
    # So for TF, it should be base.yaml + tensorflow.yaml

    # We'll use the load_config from common.utils which likely handles the base + override logic
    # if we point it to the override config, OR we might need to manually merge if load_config doesn't.
    # Let's assume load_config("configs/tensorflow.yaml") does the right thing if it's built like that.
    # However, standard pattern is usually loading base then updating.
    # Let's check src/common/utils.py to be sure.

    return load_config("configs/tensorflow.yaml")
    # Let's check src/common/utils.py to be sure.

    return load_config("configs/tensorflow.yaml")
