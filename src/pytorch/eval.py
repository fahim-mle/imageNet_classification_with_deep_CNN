import torch
import time
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
from src.pytorch.model import create_model
from src.common.paths import OUTPUTS_DIR

def load_trained_model(cfg, model_path):
    """
    Create and return a model constructed from `cfg` with weights loaded from `model_path`.
    
    The returned model is moved to the selected device (CUDA if available, else MPS if available, else CPU) and set to evaluation mode.
    
    Parameters:
        cfg (dict): Configuration used to construct the model.
        model_path (str): Filesystem path to the saved model state dict.
    
    Returns:
        model (torch.nn.Module): Model with weights loaded, moved to the device, and in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = create_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate a classification model on the provided DataLoader and compute performance and latency metrics.
    
    Returns:
        dict: Mapping with keys:
            - "accuracy" (float): Overall classification accuracy (correct / total).
            - "per_class_accuracy" (List[float]): Per-class accuracy values in the same order as confusion_matrix rows.
            - "confusion_matrix" (List[List[int]]): Square confusion matrix as a nested list (rows=true labels, columns=predicted labels).
            - "avg_inference_time" (float): Average inference time per sample in seconds.
            - "avg_batch_time" (float): Average inference time per batch in seconds.
            - "throughput" (float): Processed samples per second (total_samples / total inference time).
            - "total_samples" (int): Number of samples evaluated.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    total_inference_time = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()

            batch_latency = end_time - start_time
            total_inference_time += batch_latency
            total_batches += 1

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_batch_latency = total_inference_time / total_batches
    avg_inference_time_per_sample = total_inference_time / total
    throughput = total / total_inference_time

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    # We can derive this from confusion matrix
    # Diagonal elements / row sums
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    # Handle division by zero if any class is missing in test set
    per_class_accuracy = np.nan_to_num(per_class_accuracy)

    metrics = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": cm.tolist(),
        "avg_inference_time": avg_inference_time_per_sample,
        "avg_batch_time": avg_batch_latency,
        "throughput": throughput,
        "total_samples": total
    }

    return metrics

def setup_eval_logging(timestamp):
    """
    Configure evaluation logging to write to a timestamped file and to the console.
    
    Creates an OUTPUTS_DIR/logs directory if missing, configures the root logger to write INFO-level messages to a file named `eval_{timestamp}.log` with a timestamped message format, resets existing handlers, and adds an INFO-level console (stream) handler.
    
    Parameters:
        timestamp (str): Identifier (typically a timestamp) appended to the log filename.
    """
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / f"eval_{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True # Reset any existing handlers
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)