import tensorflow as tf
import os
import json
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.paths import OUTPUTS_DIR
from src.tensorflow.data import load_dataset, prepare_tf_dataset
from src.tensorflow.utils import setup_logging

def evaluate_model(cfg, model_path):
    """
    Evaluate saved model on test dataset
    """
    logger = setup_logging(cfg)
    logger.info(f"Evaluating model: {model_path}")

    # 1. Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load Test Dataset
    _, _, test_ds, class_names = load_dataset(cfg)
    if test_ds is None:
        logger.error("Failed to load test dataset.")
        return

    test_ds = prepare_tf_dataset(test_ds, cfg, training=False)

    # 3. Compute Metrics
    logger.info("Running inference...")

    y_true = []
    y_pred = []
    latencies = []

    start_time = time.time()

    for images, labels in test_ds:
        batch_start = time.time()
        preds = model.predict(images, verbose=0)
        batch_end = time.time()

        latencies.append(batch_end - batch_start)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    total_time = time.time() - start_time
    num_samples = len(y_true)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    avg_latency = np.mean(latencies)
    throughput = num_samples / total_time

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Latency: {avg_latency:.4f} s/batch")
    logger.info(f"Throughput: {throughput:.2f} samples/s")

    # 4. Save Results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics", "tf", timestamp)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save metrics.json
    metrics_data = {
        "accuracy": float(accuracy),
        "avg_latency": float(avg_latency),
        "throughput": float(throughput),
        "per_class_accuracy": {}
    }

    # Per-class accuracy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i, name in enumerate(class_names):
        acc = cm_normalized[i, i]
        metrics_data["per_class_accuracy"][name] = float(acc)

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"))
    plt.close()

    # Log to file
    log_file = os.path.join(OUTPUTS_DIR, "tf_logs", f"eval_{timestamp}.log")
    with open(log_file, "w") as f:
        f.write(f"Evaluation Results ({timestamp})\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Throughput: {throughput:.2f} samples/s\n")
