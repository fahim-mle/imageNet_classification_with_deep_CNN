# TensorFlow Core Phase 1 - Project-Aware Instruction Set

*(based on your current project structure)*

## 🟧 TASK 0 — New Branch

Create new branch:

```bash
git checkout -b feature/tf-core
```

This branch must contain all TF-specific files.

## 🟧 TASK 1 — Add TensorFlow Config File

Create: `configs/tensorflow.yaml`

This file extends `base.yaml` and matches the pattern already used by `pytorch.yaml`.

Content scaffold:

```yaml
framework: "tensorflow"

dataset:
  image_size: [128, 128]     # override allowed
  shuffle_buffer: 2000
  num_parallel_calls: "auto"

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  mixed_precision: true

model:
  architecture: "resnet50"   # placeholder
  pretrained: true
  num_classes: 10

optimizer:
  name: "adam"
  weight_decay: 0.0001

logging:
  save_dir: "outputs/tf_logs"
  checkpoint_dir: "models/tf"
```

Your agent must NOT mix this with PyTorch logic.

## 🟧 TASK 2 — Create TensorFlow Directory Structure

Everything must live under `src/tensorflow/`:

```
src/
    tensorflow/
        __init__.py
        data.py
        transforms.py
        model.py
        train.py
        eval.py
        utils.py
```

Your PyTorch code under `src/pytorch/` must remain untouched.

## 🟧 TASK 3 — Create TensorFlow Scripts Folder

Like PyTorch has its own script folder (`scripts/pytorch/`), TF must also have its own:

```
scripts/
    tensorflow/
        __init__.py
        run_train.py        # later
        run_eval.py         # later
```

Right now these can be skeleton-only.

## 🟧 TASK 4 — Update main.py Cleanly

Modify `main.py` to respect frameworks:

```python
if cfg.framework == "tensorflow":
    from src.tensorflow.train import train_model
    train_model(cfg)
```

This must NOT affect existing PyTorch flow.

## 🟧 TASK 5 — TF Utility Skeletons

Inside `src/tensorflow/utils.py`, create placeholder functions:

### 1. setup_device()

```python
def setup_device():
    """
    Detect GPU via tf.config.list_physical_devices()
    Return "GPU" or "CPU"
    No training logic
    """
    pass
```

### 2. setup_logging(cfg)

```python
def setup_logging(cfg):
    """
    Create directories:
    - outputs/tf_logs/
    - models/tf/

    Return logger (Python logging)
    """
    pass
```

### 3. load_tf_config()

```python
def load_tf_config():
    """
    Merge base.yaml + tensorflow.yaml
    Return config dict/object
    """
    pass
```

## 🟧 TASK 6 — TF Dataset Pipeline Skeleton

Inside `src/tensorflow/data.py`, create EMPTY (signature-only) functions:

### load_dataset(cfg)

```python
def load_dataset(cfg):
    """
    Load train/val/test folders from data/processed/
    Return placeholders (no tf.data operations yet)
    """
    pass
```

### prepare_tf_dataset(ds, cfg)

```python
def prepare_tf_dataset(ds, cfg):
    """
    Placeholder for batching/shuffling/prefetching
    """
    pass
```

### get_class_names(cfg)

```python
def get_class_names(cfg):
    """
    Scan data/processed/train/ and return folder names
    """
    pass
```

**MUST NOT import PyTorch utilities**

## 🟧 TASK 7 — TF Model Definition Scaffold

Inside `src/tensorflow/model.py`:

```python
def create_model(cfg):
    """
    Placeholder for TensorFlow/Keras model creation.
    Will support:
    - keras.applications.ResNet / EfficientNet / MobileNet
    - custom classification head
    - mixed precision (optional)
    """
    raise NotImplementedError
```

No code yet.

## 🟧 TASK 8 — Training Skeleton

Inside `src/tensorflow/train.py`:

Implement placeholders:

### compile_model(model, cfg)

```python
def compile_model(model, cfg):
    """
    Docstring only
    """
    pass
```

### train_model(cfg)

```python
def train_model(cfg):
    """
    Docstring describing:
    - Load config
    - Prepare datasets
    - Create model
    - Compile with optimizer/loss
    - Setup callbacks
    - Save checkpoints
    - Log metrics
    """
    pass
```

No actual code yet.

## 🟧 TASK 9 — Evaluation Skeleton

Inside `src/tensorflow/eval.py`:

Create:

### evaluate_model(cfg, model_path)

```python
def evaluate_model(cfg, model_path):
    """
    Docstring only
    Describes future responsibilities:
    - Load saved .h5 or Checkpoint
    - Load test dataset
    - Compute metrics
    - Compute confusion matrix
    - Save results under: outputs/metrics/tf/{timestamp}/
    - Generate PNG plots later
    """
    pass
```

## 🟧 TASK 10 — Setup Output Directories

Ensure these exist:

```
outputs/
    tf_logs/
    metrics/
        tf/
models/
    tf/
```

**DO NOT put TF outputs into PyTorch folders.**

## 🟧 TASK 11 — Write Documentation

Add this file: `docs/instructions/for_TensorFlow/01_TF_core_implementation.md`

Content should document:

- How configs are structured
- How TF directory structure works
- What Phase 2 + Phase 3 will implement
- Example command to run TF core
- Difference between TF and PyTorch structure
