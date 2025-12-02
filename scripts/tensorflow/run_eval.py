import sys
import os
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.common.utils import load_config
from src.common.paths import MODELS_DIR
from src.tensorflow.eval import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run TensorFlow Evaluation")
    parser.add_argument("--model_path", type=str, help="Path to model file", default=None)
    args = parser.parse_args()

    cfg = load_config("configs/tensorflow.yaml")

    model_path = args.model_path
    if model_path is None:
        # Default to best model
        model_path = os.path.join(MODELS_DIR, "tf", "best_model.keras")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    evaluate_model(cfg, model_path)

if __name__ == "__main__":
    main()
