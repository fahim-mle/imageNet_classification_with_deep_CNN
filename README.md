# ImageNet Classification with Deep CNNs

This project implements and compares multiple deep-learning approaches for image classification using the Animals-10 dataset. It includes three parallel models:

- A custom PyTorch CNN
- A custom TensorFlow/Keras CNN
- A Hugging Face pretrained model for transfer-learning comparison

The main goal is to understand architectural differences, training behavior, and performance across frameworks.

## Project Structure

(placeholder — will be updated later)

## Setup

(placeholder)

## Training

### PyTorch Implementation

To run the PyTorch pipeline, use the following commands:

**Training:**
```bash
uv run python main.py --framework pytorch --train
```

**Evaluation:**
```bash
uv run python main.py --framework pytorch --eval
```

This will:
1. Load the configuration from `configs/pytorch.yaml`.
2. Execute the requested mode (train or eval).
3. Save logs to `outputs/logs/` and artifacts to `outputs/metrics/` (for eval).

### Configuration

You can modify training parameters in `configs/pytorch.yaml`.

## Evaluation

(placeholder)

## Models

(placeholder)

## Dataset

(placeholder)

## Results

(placeholder)
