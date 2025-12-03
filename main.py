import argparse
import sys
import os
from src.common.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="ImageNet Classification with Deep CNNs")
    parser.add_argument("--framework", type=str, default="pytorch", choices=["pytorch", "tensorflow", "huggingface"], help="Deep learning framework to use")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")

    args = parser.parse_args()

    if args.framework == "pytorch":
        cfg = load_config("configs/pytorch.yaml")

        if args.train:
            print("Starting PyTorch Training...")
            from src.pytorch.train import main_pytorch
            main_pytorch()

        if args.eval:
            print("Starting PyTorch Evaluation...")
            from scripts.pytorch.run_eval import main_eval
            main_eval()

        if not args.train and not args.eval:
            print("No mode selected. Use --train or --eval.")
            parser.print_help()

    elif args.framework == "tensorflow":
        cfg = load_config("configs/tensorflow.yaml")

        if args.train:
            print("Starting TensorFlow Training...")
            from src.tensorflow.train import train_model
            train_model(cfg)

        if args.eval:
            print("Starting TensorFlow Evaluation...")
            from src.common.paths import MODELS_DIR
            from src.tensorflow.eval import evaluate_model

            # Default to best model
            model_path = os.path.join(MODELS_DIR, "tf", "best_model.keras")
            if os.path.exists(model_path):
                evaluate_model(cfg, model_path)
            else:
                print(f"No model found at {model_path}. Please train first.")

    else:
        print(f"Framework {args.framework} not implemented yet.")

if __name__ == "__main__":
    main()
