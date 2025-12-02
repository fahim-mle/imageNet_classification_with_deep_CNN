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

To run the PyTorch training pipeline:

```bash
uv run python main.py
```

This will:
1. Load the configuration from `configs/pytorch.yaml`.
2. Download the ResNet18 model (if not already cached).
3. Train the model on the "Animals-10" dataset.
4. Save logs to `outputs/logs/` and checkpoints to `models/pytorch/`.

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
