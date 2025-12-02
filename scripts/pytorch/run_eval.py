import os
import time
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.common.utils import load_config
from src.pytorch.data import get_dataloaders
from src.pytorch.eval import load_trained_model, evaluate_model, setup_eval_logging
from src.common.paths import OUTPUTS_DIR, MODELS_DIR

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main_eval():
    # Setup logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_eval_logging(timestamp)

    # Load config
    cfg = load_config("configs/pytorch.yaml")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Data
    _, _, test_loader = get_dataloaders(cfg)

    # We need class names for the confusion matrix.
    # ImageFolder provides `classes` attribute but our get_dataloaders returns loaders.
    # We can access the dataset from the loader.
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else \
                  test_loader.dataset.subset.dataset.classes if hasattr(test_loader.dataset, 'subset') else \
                  [str(i) for i in range(10)] # Fallback

    # Create metrics output dir
    metrics_dir = OUTPUTS_DIR / "metrics" / f"eval_{timestamp}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = ["best_model.pt", "last_model.pt"]

    for model_file in models_to_eval:
        model_path = MODELS_DIR / "pytorch" / model_file
        if not model_path.exists():
            print(f"Model {model_file} not found at {model_path}. Skipping.")
            continue

        print(f"Evaluating {model_file}...")
        model = load_trained_model(cfg, str(model_path))

        metrics = evaluate_model(model, test_loader, device)

        # Save metrics
        metrics_file = metrics_dir / f"{model_file.split('.')[0]}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Plot confusion matrix
        cm_file = metrics_dir / f"{model_file.split('.')[0]}_confusion_matrix.png"
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_file)

        print(f"Evaluation for {model_file} complete. Metrics saved to {metrics_dir}")

if __name__ == "__main__":
    main_eval()
