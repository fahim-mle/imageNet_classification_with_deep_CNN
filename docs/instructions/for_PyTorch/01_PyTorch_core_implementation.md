# Instructions — PyTorch Core Implementation

## Goal

Implement the full PyTorch training pipeline using the project skeleton and config system.

The agent must follow the tasks below in order.

---

## 🟦 Phase 1 — Setup

### Task 1 — Create Framework Config

Generate file:

```
configs/pytorch.yaml
```

With keys:

```yaml
framework: pytorch
model:
  name: resnet18
  pretrained: true
optimizer:
  name: adam
  weight_decay: 1e-4
loss:
  name: cross_entropy
training:
  batch_size: override base
  epochs: override base
```

Values must merge with base.yaml.

---

## 🟦 Phase 2 — Data Pipeline

### Task 2 — Implement Transforms

**File:** `src/pytorch/transforms.py`

Create function:

```python
get_transforms(image_size)
```

Returning train/val/test transforms.

### Task 3 — Dataset + Dataloader

**File:** `src/pytorch/data.py`

Implement:

```python
def get_dataloaders(cfg):
    return train_loader, val_loader, test_loader
```

Uses:

- `datasets.ImageFolder`
- Processed paths from config
- Transforms from Task 2
- Batch size from cfg

---

## 🟦 Phase 3 — Model Architecture

### Task 4 — Model Loader

**File:** `src/pytorch/model.py`

Implement function:

```python
def create_model(cfg):
```

- Loads resnet18 (pretrained optional)
- Replaces final layer based on num classes
- Moves to device

---

## 🟦 Phase 4 — Training Pipeline

### Task 5 — Training Loop

**File:** `src/pytorch/train.py`

Implement:

```python
def train_model(cfg, model, loaders):
```

- Optimizer from cfg
- Loss fn from cfg
- Device-aware
- Best model saving
- Log metrics to outputs/logs

No HF Trainer, no magic.

### Task 6 — Evaluation

**File:** `src/pytorch/eval.py`

Implement:

```python
def evaluate(model, loader, device):
```

Returns accuracy, loss.

---

## 🟦 Phase 5 — Entry Point

### Task 7 — Main Script

Edit `main.py`:

```python
if config.framework == "pytorch":
    run:
    from src.pytorch.train import main_pytorch
    main_pytorch()
