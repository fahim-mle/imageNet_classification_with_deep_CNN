import os
from pathlib import Path

# Project root is 2 levels up from this file (src/common/paths.py -> src/common -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

def get_config_path(config_name="base.yaml"):
    return CONFIGS_DIR / config_name
