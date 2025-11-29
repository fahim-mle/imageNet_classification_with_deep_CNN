import os
import yaml
import shutil
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.common.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, get_config_path
from src.common.utils import setup_logging, seed_everything

logger = setup_logging("preprocess_data.log")

def load_config():
    config_path = get_config_path()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_image(src_path, dest_path, size):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(dest_path)
            return True
    except Exception as e:
        logger.warning(f"Failed to process {src_path}: {e}")
        return False

def preprocess_data():
    config = load_config()
    seed_everything(config["dataset"]["seed"])

    image_size = tuple(config["dataset"]["image_size"])
    split_ratios = config["dataset"]["split_ratios"]

    # The Animals-10 dataset structure usually has 'raw/animals/animals/...' or similar.
    # We need to find the root folder containing class folders.
    # Let's assume standard structure: raw/class_name/image.jpg
    # Or raw/dataset_name/class_name/image.jpg

    # Simple heuristic to find class folders
    # We look for directories that contain images

    class_dirs = []
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        # If this directory contains images, and its parent is not the raw dir itself (unless flat structure)
        # Actually, let's just look for subdirectories in RAW_DATA_DIR that are not hidden
        # But Animals-10 might be nested.
        # Let's just find all directories that contain images.

        has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
        if has_images:
            # Check if it's a leaf node (class folder)
            # We treat the folder name as the class name
            class_dirs.append(Path(root))

    if not class_dirs:
        logger.error("No image directories found in raw data!")
        return

    logger.info(f"Found {len(class_dirs)} class directories.")

    # Prepare processed directories
    for split in ['train', 'val', 'test']:
        (PROCESSED_DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    summary = {"classes": [], "counts": {"train": 0, "val": 0, "test": 0}}

    for class_dir in class_dirs:
        class_name = class_dir.name
        summary["classes"].append(class_name)

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Split
        train_val, test = train_test_split(images, test_size=split_ratios['test'], random_state=config["dataset"]["seed"])
        train, val = train_test_split(train_val, test_size=split_ratios['val'] / (1 - split_ratios['test']), random_state=config["dataset"]["seed"])

        splits = {'train': train, 'val': val, 'test': test}

        logger.info(f"Processing class: {class_name} ({len(images)} images)")

        for split, split_images in splits.items():
            split_dir = PROCESSED_DATA_DIR / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_name in tqdm(split_images, desc=f"{class_name} [{split}]", leave=False):
                src = class_dir / img_name
                dest = split_dir / img_name
                if process_image(src, dest, image_size):
                    summary["counts"][split] += 1

    with open(PROCESSED_DATA_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Preprocessing complete. Summary saved.")

if __name__ == "__main__":
    preprocess_data()
