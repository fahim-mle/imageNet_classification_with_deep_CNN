import os
import yaml
import json
from dotenv import load_dotenv
from src.common.paths import RAW_DATA_DIR, get_config_path
from src.common.utils import setup_logging

load_dotenv()

# Debug: Check if Kaggle credentials are present
if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_TOKEN" in os.environ:
    print("Mapping KAGGLE_API_TOKEN to KAGGLE_KEY...")
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    print("Error: KAGGLE_USERNAME and/or KAGGLE_KEY not found in environment variables.")
    print("Available keys:", [k for k in os.environ.keys() if "KAGGLE" in k])
else:
    print(f"Kaggle credentials found for user: {os.environ['KAGGLE_USERNAME']}")

logger = setup_logging("download_data.log")

def load_config():
    config_path = get_config_path()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_data():
    config = load_config()
    dataset_name = config["dataset"]["name"]

    logger.info(f"Downloading dataset: {dataset_name}...")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=RAW_DATA_DIR, unzip=True)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_data()
