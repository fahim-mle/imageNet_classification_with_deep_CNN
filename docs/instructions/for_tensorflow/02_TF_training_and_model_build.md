# TensorFlow Training and Model Building

This document details the implementation of the TensorFlow training pipeline, model building, and evaluation.

## 1. Data Pipeline

**File:** `src/tensorflow/data.py`

The data pipeline uses `tf.data` for efficient loading and preprocessing.

- **Loading:** `load_dataset(cfg)` loads images from `data/processed/{train,val,test}` using `tf.keras.preprocessing.image_dataset_from_directory`.
- **Preprocessing:** `prepare_tf_dataset(ds, cfg, training)` applies transforms, batching, shuffling (training only), and prefetching.
- **Transforms:** `src/tensorflow/transforms.py` handles resizing, normalization, and augmentation (RandomFlip, RandomRotation).

## 2. Model Architecture

**File:** `src/tensorflow/model.py`

The model is built using `tf.keras.applications` backbones (e.g., ResNet50) with a custom classification head.

- **Backbone:** Loaded with `include_top=False` and `pooling="avg"`.
- **Head:** Dense(256, ReLU) -> Dropout(0.3) -> Dense(num_classes, Softmax).
- **Mixed Precision:** Supported via `cfg.training.mixed_precision`.

## 3. Training Loop

**File:** `src/tensorflow/train.py`

The training loop manages the model lifecycle.

- **Compilation:** Adam optimizer, Sparse Categorical Crossentropy loss, Accuracy metric.
- **Callbacks:**
    - `ModelCheckpoint`: Saves best model to `models/tf/best_model.keras`.
    - `CSVLogger`: Logs training history to `outputs/tf_logs/train_{timestamp}.csv`.
    - `EarlyStopping`: Stops training if validation loss doesn't improve.
- **Execution:** Uses `model.fit()`.

## 4. Evaluation

**File:** `src/tensorflow/eval.py`

Evaluation computes metrics on the test dataset.

- **Metrics:** Overall Accuracy, Per-class Accuracy, Confusion Matrix, Latency, Throughput.
- **Outputs:**
    - `outputs/metrics/tf/{timestamp}/metrics.json`
    - `outputs/metrics/tf/{timestamp}/confusion_matrix.png`
    - `outputs/tf_logs/eval_{timestamp}.log`

## 5. Usage

### Training

```bash
python scripts/tensorflow/run_train.py
# OR
python main.py --framework tensorflow --train
```

### Evaluation

```bash
python scripts/tensorflow/run_eval.py --model_path models/tf/best_model.keras
# OR
python main.py --framework tensorflow --eval
```

## 6. Project Isolation

All TensorFlow code is contained within `src/tensorflow/` and `scripts/tensorflow/`. Outputs are directed to `outputs/tf_logs/`, `outputs/metrics/tf/`, and `models/tf/`, ensuring no conflict with PyTorch artifacts.
