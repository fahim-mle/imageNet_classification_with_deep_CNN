import argparse
import sys
from src.common.utils import load_config
import os

def main():
    """
    Parse command-line arguments and dispatch to the selected framework's training or evaluation entry points.
    
    When the --framework is "pytorch" the function loads the PyTorch configuration and, depending on flags, starts training (calls src.pytorch.train.main_pytorch) and/or evaluation (calls scripts.pytorch.run_eval.main_eval). If neither --train nor --eval is provided it prints a message and shows usage help. For other framework values it prints that the framework is not implemented.
    """
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

    else:
        print(f"Framework {args.framework} not implemented yet.")

if __name__ == "__main__":
    main()