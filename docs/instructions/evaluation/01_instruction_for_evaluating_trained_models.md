# Model Evaluation Instructions

## ✅ Updated Task Breakdown (Matches Your Project & Assignment)

### A. Model Evaluation (All Models)

For PyTorch ResNet18, TensorFlow MobileNetV2, and Baseline CNN:

#### Steps

1. Load the model(s)
2. Load dataset (train/val/test)
3. Evaluate on all splits

#### Metrics to Calculate

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Per-class accuracy
- AUC + ROC (multi-class one-vs-rest)

#### Output Structure

Save metrics under:

```
outputs/metrics/<model_name>_eval_<timestamp>/
    <model_name>_metrics.json
    <model_name>_confusion_matrix.png
```

> **Note:** Your existing folder style remains intact (best/last etc.).

---

### B. Visual Outputs (Plots/Graphs)

All plots go under:

```
outputs/figures/<model_name>_eval_<timestamp>/
```

#### Plots to generate

- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png`
- `sample_predictions.png`
- `per_class_accuracy.png` (optional but recommended)

#### Quality Requirements

- High-quality (300 dpi)
- Consistent color styling
- Model name embedded in plot title

---

### C. Prediction Outputs

Keep storing any raw predictions under:

```
outputs/predictions/
```

#### Data to Store

- Predicted labels
- True labels
- Probabilities
- Optional CSV: `predictions_<timestamp>.csv`

---

### D. Logging

All eval operations must emit a log file under:

```
outputs/logs/<model_name>_eval_<timestamp>.log
```

---

## Implementation Notes

- Follow the existing folder structure patterns
- Maintain consistency across all model evaluations
- Ensure timestamped outputs for traceability
- Use standardized naming conventions for all output files

## Implementation Status

✅ **Implemented**: `scripts/evaluate_all_models.py` handles all the above steps.
