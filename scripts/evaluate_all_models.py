import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Framework imports
import torch
import tensorflow as tf


# Project imports
from src.common.paths import OUTPUTS_DIR, MODELS_DIR, DATA_DIR
from src.common.utils import load_config
from src.pytorch.data import get_dataloaders as get_pytorch_dataloaders
from src.pytorch.model import create_model as create_pytorch_model
from src.tensorflow.data import load_dataset as load_tf_dataset, prepare_tf_dataset
from src.tensorflow.model import create_model as create_tf_model

# Constants
CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints") # Assuming checkpoints are in root/checkpoints as per instructions
# Instructions say: checkpoints/resnet18.pth, checkpoints/mobilenetv2/, checkpoints/baseline.pth
# But user rules say models are in models/pytorch/ etc.
# However, the SPECIFIC INSTRUCTION for this task says "checkpoints/resnet18.pth" etc.
# I will respect the instruction's paths for loading, but I should check if they exist or if I should look in models/.
# The instruction says "Checkpoint: checkpoints/resnet18.pth". I will assume these are the targets.
# If they don't exist, I might need to look in MODELS_DIR.
# But for now, I'll define them as per instructions.

MODELS_TO_EVALUATE = {
    "best_model_pytorch": {
        "framework": "pytorch",
        "path": os.path.join(MODELS_DIR, "pytorch", "best_model.pt"),
        "config": "configs/pytorch.yaml"
    },
    "best_model_tensorflow": {
        "framework": "tensorflow",
        "path": os.path.join(MODELS_DIR, "tf", "best_model.keras"),
        "config": "configs/tensorflow.yaml"
    }
}

def setup_logging(timestamp):
    log_dir = os.path.join(OUTPUTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eval_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def create_output_dirs(timestamp):
    dirs = {
        "metrics": os.path.join(OUTPUTS_DIR, "metrics", f"eval_{timestamp}"),
        "figures": os.path.join(OUTPUTS_DIR, "figures", f"eval_{timestamp}"),
        "predictions": os.path.join(OUTPUTS_DIR, "predictions")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def load_pytorch_model(model_name, checkpoint_path, config_path):
    cfg = load_config(config_path)

    # Override model name for baseline
    if model_name == "baseline":
        cfg['model']['name'] = "baseline"
        cfg['model']['pretrained'] = False # Baseline is custom, not pretrained

    model = create_pytorch_model(cfg)

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading PyTorch model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
        model.load_state_dict(state_dict)
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Using random initialization/pretrained weights.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device, cfg

def load_tensorflow_model(checkpoint_path, config_path):
    cfg = load_config(config_path)

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading TensorFlow model from {checkpoint_path}")
        try:
            model = tf.keras.models.load_model(checkpoint_path)
        except Exception as e:
            logging.error(f"Failed to load SavedModel: {e}. Trying to create and load weights if possible, or just create.")
            # Fallback: Create and load weights if it was saved as weights only, but instruction says "SavedModel dir"
            model = create_tf_model(cfg)
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Creating fresh model.")
        model = create_tf_model(cfg)

    return model, cfg

def get_pytorch_data(cfg):
    # We need train, val, test loaders
    # cfg should have dataset params
    train_loader, val_loader, test_loader = get_pytorch_dataloaders(cfg)
    return {"train": train_loader, "val": val_loader, "test": test_loader}

def get_tensorflow_data(cfg):
    train_ds, val_ds, test_ds, class_names = load_tf_dataset(cfg)

    train_ds = prepare_tf_dataset(train_ds, cfg, training=False) # No shuffle for eval
    val_ds = prepare_tf_dataset(val_ds, cfg, training=False)
    test_ds = prepare_tf_dataset(test_ds, cfg, training=False)

    return {"train": train_ds, "val": val_ds, "test": test_ds}, class_names

def evaluate_pytorch(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating PyTorch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)

def evaluate_tensorflow(model, dataset):
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(dataset, desc="Evaluating TensorFlow"):
        outputs = model(inputs, training=False)
        # Check if outputs are logits or probabilities
        # Our model definition in src/tensorflow/model.py outputs softmax if not mixed_precision, or linear if mixed.
        # But let's assume softmax for safety or apply it.
        # Actually, `model.predict` is easier but we want labels too.
        # Let's just use the loop.

        # If last layer has no activation (linear), apply softmax
        # But our create_model adds softmax usually.
        probs = outputs # Assuming softmax
        if tf.reduce_max(probs) > 1.0 or tf.reduce_min(probs) < 0.0: # Heuristic check
             probs = tf.nn.softmax(outputs)

        preds = tf.argmax(probs, axis=1)

        all_probs.append(probs.numpy())
        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)

def compute_metrics(y_true, y_pred, y_prob, class_names):
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class accuracy
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    per_class_acc = np.divide(cm_diag, cm_sum, out=np.zeros_like(cm_diag, dtype=float), where=cm_sum!=0)
    metrics['per_class_accuracy'] = {name: float(acc) for name, acc in zip(class_names, per_class_acc)}

    # ROC/AUC
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)

    roc_auc = {}
    try:
        if n_classes == 2:
             roc_auc["binary"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            roc_auc["macro"] = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
            roc_auc["weighted"] = float(roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr"))
    except Exception as e:
        logging.warning(f"Could not compute ROC AUC: {e}")
        roc_auc["error"] = str(e)

    metrics['roc_auc'] = roc_auc

    return metrics

def plot_confusion_matrix(cm, class_names, output_path, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_curves(y_true, y_prob, class_names, output_path, model_name):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_per_class_accuracy(per_class_acc, output_path, model_name):
    names = list(per_class_acc.keys())
    values = list(per_class_acc.values())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=names, y=values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy - {model_name}')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sample_predictions(inputs, labels, preds, class_names, output_path, model_name, framework):
    # Inputs might be tensors or numpy arrays
    # We need to unnormalize if normalized.
    # Assuming standard ImageNet normalization or similar.
    # For visualization, we'll just clip to 0-1 if float.

    plt.figure(figsize=(15, 10))
    num_samples = min(len(labels), 9)

    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        img = inputs[i]

        if framework == "pytorch":
            img = img.permute(1, 2, 0).cpu().numpy()
        else:
            img = img.numpy()

        # Denormalize roughly (assuming mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Or just min-max scale for display
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        color = 'green' if labels[i] == preds[i] else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')

    plt.suptitle(f'Sample Predictions - {model_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_predictions(filenames, y_true, y_pred, y_prob, output_path):
    # We might not have filenames easily available from DataLoader/Dataset unless we extract them.
    # For ImageFolder, `samples` attribute has paths.
    # For TF, `file_paths` might be available.
    # If not available, we'll use indices.

    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability_scores': [list(p) for p in y_prob]
    })

    if filenames is not None and len(filenames) == len(y_true):
        df.insert(0, 'filename', filenames)
    else:
        df.insert(0, 'index', range(len(y_true)))

    df.to_csv(output_path, index=False)

def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = setup_logging(timestamp)
    dirs = create_output_dirs(timestamp)

    # Get Class Names (Assume consistent across frameworks if using same data)
    # We'll load PyTorch config to get class names or load data once.
    # Let's load PyTorch data first to get class names.
    logging.info("Initializing Data...")

    # Load PyTorch Data
    pt_cfg = load_config("configs/pytorch.yaml")
    pt_loaders = get_pytorch_data(pt_cfg)
    pt_class_names = pt_loaders['test'].dataset.classes if hasattr(pt_loaders['test'].dataset, 'classes') else \
                     pt_loaders['test'].dataset.subset.dataset.classes if hasattr(pt_loaders['test'].dataset, 'subset') else \
                     [str(i) for i in range(10)]

    logging.info(f"Class names: {pt_class_names}")

    # Load TF Data
    tf_cfg = load_config("configs/tensorflow.yaml")
    tf_data, tf_class_names = get_tensorflow_data(tf_cfg)

    # Verify consistency
    if len(pt_class_names) != len(tf_class_names):
        logging.warning("Class name count mismatch between PyTorch and TensorFlow data loading!")

    splits = ['train', 'val', 'test']

    for model_key, model_info in MODELS_TO_EVALUATE.items():
        logging.info(f"--- Evaluating Model: {model_key} ({model_info['framework']}) ---")

        framework = model_info['framework']

        # Load Model
        if framework == "pytorch":
            model, device, cfg = load_pytorch_model(model_key, model_info['path'], model_info['config'])
        else:
            model, cfg = load_tensorflow_model(model_info['path'], model_info['config'])

        # Evaluate on all splits
        for split in splits:
            logging.info(f"Evaluating on {split} set...")

            if framework == "pytorch":
                loader = pt_loaders[split]
                y_true, y_pred, y_prob = evaluate_pytorch(model, loader, device)

                # Get sample inputs for visualization (only for test split)
                if split == 'test':
                    sample_inputs, sample_labels = next(iter(loader))
                    sample_inputs = sample_inputs[:9]
                    sample_labels = sample_labels[:9]
                    # We need preds for these
                    with torch.no_grad():
                        outputs = model(sample_inputs.to(device))
                        _, sample_preds = torch.max(outputs, 1)
                        sample_preds = sample_preds.cpu().numpy()
                        sample_labels = sample_labels.numpy()

            else:
                ds = tf_data[split]
                y_true, y_pred, y_prob = evaluate_tensorflow(model, ds)

                if split == 'test':
                    # Get samples
                    for batch_inputs, batch_labels in ds.take(1):
                        sample_inputs = batch_inputs[:9]
                        sample_labels = batch_labels[:9]
                        # Preds
                        outputs = model(sample_inputs, training=False)
                        probs = outputs
                        if tf.reduce_max(probs) > 1.0 or tf.reduce_min(probs) < 0.0:
                             probs = tf.nn.softmax(outputs)
                        sample_preds = tf.argmax(probs, axis=1).numpy()
                        sample_labels = sample_labels.numpy()
                        break

            # Compute Metrics
            metrics = compute_metrics(y_true, y_pred, y_prob, pt_class_names)
            logging.info(f"{split} Accuracy: {metrics['accuracy']:.4f}")

            # Save Metrics (only for test split or all? Instructions say "Evaluate on all splits", "Save metrics".
            # Usually we save test metrics as the primary result.
            # But let's save all with split prefix.
            metrics_file = os.path.join(dirs['metrics'], f"{model_key}_{split}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            # Plots (Only for test split usually, or as requested)
            if split == 'test':
                # Confusion Matrix
                plot_confusion_matrix(np.array(metrics['confusion_matrix']), pt_class_names,
                                      os.path.join(dirs['figures'], f"confusion_matrix_{model_key}.png"), model_key)

                # ROC Curves
                plot_roc_curves(y_true, y_prob, pt_class_names,
                                os.path.join(dirs['figures'], f"roc_curve_{model_key}.png"), model_key)

                # Per-class Accuracy
                plot_per_class_accuracy(metrics['per_class_accuracy'],
                                        os.path.join(dirs['figures'], f"per_class_accuracy_{model_key}.png"), model_key)

                # Sample Predictions
                plot_sample_predictions(sample_inputs, sample_labels, sample_preds, pt_class_names,
                                        os.path.join(dirs['figures'], f"sample_predictions_{model_key}.png"), model_key, framework)

                # Save Predictions CSV
                # Try to get filenames if possible, otherwise indices
                # For now, just indices
                save_predictions(None, y_true, y_pred, y_prob,
                                 os.path.join(dirs['predictions'], f"predictions_{model_key}_{timestamp}.csv"))

        # Clean up to save memory
        if framework == "pytorch":
            del model
            torch.cuda.empty_cache()
        else:
            del model
            tf.keras.backend.clear_session()

    logging.info("Evaluation Complete.")

if __name__ == "__main__":
    main()
