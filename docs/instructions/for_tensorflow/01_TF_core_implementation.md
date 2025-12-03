# TensorFlow Core Implementation

This document outlines the structure and usage of the TensorFlow implementation for the ImageNet classification project.

## Configuration

The TensorFlow configuration is defined in `configs/tensorflow.yaml`. It extends `configs/base.yaml` and overrides specific settings for TensorFlow.

Key sections:
- `framework`: Set to "tensorflow"
- `dataset`: TensorFlow-specific dataset parameters (e.g., `shuffle_buffer`, `num_parallel_calls`)
- `training`: Training hyperparameters (batch size, epochs, learning rate)
- `model`: Model architecture and settings
- `logging`: Paths for logs and checkpoints

## Directory Structure

The TensorFlow implementation resides in `src/tensorflow/`, mirroring the PyTorch structure:

```
src/
    tensorflow/
        __init__.py
        data.py       # Dataset loading and pipeline (tf.data)
        transforms.py # Data augmentations
        model.py      # Model definition (Keras)
        train.py      # Training loop
        eval.py       # Evaluation logic
        utils.py      # TensorFlow-specific utilities
```

Scripts are located in `scripts/tensorflow/`.

## Usage

To run TensorFlow training:

```bash
python main.py --framework tensorflow --train
```

## Implementation Status

### Phase 1: Configuration & Structure (Completed)
- Created `configs/tensorflow.yaml`
- Created directory structure
- Setup output directories

### Phase 2: Core Skeletons (Completed)
- Implemented `src/tensorflow/utils.py`
- Created placeholders for `data.py`, `model.py`, `train.py`, `eval.py`

### Future Phases
- **Phase 3**: Implement Data Pipeline (`data.py`)
- **Phase 4**: Implement Model (`model.py`)
- **Phase 5**: Implement Training Loop (`train.py`)
- **Phase 6**: Implement Evaluation (`eval.py`)

## Differences from PyTorch

- **Data Loading**: Uses `tf.data.Dataset` instead of `torch.utils.data.DataLoader`.
- **Model**: Uses `tf.keras.Model` instead of `torch.nn.Module`.
- **Training**: Will use a custom training loop or `model.fit()` (TBD in implementation).
- **Logging**: Uses TensorFlow-specific logging paths.
