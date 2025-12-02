# INSTRUCTION SET: PYTORCH MODEL EVALUATION + METRICS LOGGING

Your agent must implement a full evaluation workflow for saved models (`best_model.pt`, `last_model.pt`).
All code must follow existing project structure and configs.

## TASK 1 — Create `src/pytorch/eval.py`

Agent must implement a module with the following functions:

### 1. `load_trained_model(cfg, model_path)`

**Requirements:**

- Reconstruct model using `create_model(cfg)`
- Load weights into model with `load_state_dict`
- Move model to correct device
- Set model to `eval()` mode
- Return loaded model

### 2. `evaluate_model(model, test_loader, device)`

Agent must:

- Iterate over full `test_loader`
- Collect:
  - Total samples
  - Total correct predictions
  - Per-class accuracy
  - Confusion matrix
  - Inference latency per image
  - Average batch latency
  - Model throughput (images / second)

**Return all metrics in a dictionary format:**

```json
{
  "accuracy": ...,
  "per_class_accuracy": {...},
  "confusion_matrix": tensor or array,
  "avg_inference_time": ...,
  "avg_batch_time": ...,
  "throughput": ...
}
```

## TASK 2 — Create standalone evaluation script

**File:** `scripts/run_eval.py`

Agent must implement:

- Load config (`pytorch.yaml` merged with `base`)
- Detect device (`cuda > mps > cpu`)
- Load test dataloader via existing `get_dataloaders(cfg)`
- Load `best_model.pt`
- Load `last_model.pt`
- Evaluate both models using the functions created above
- Log results to:

  ```
  outputs/metrics/eval_{timestamp}/
      best_model_metrics.json
      last_model_metrics.json
      confusion_matrix.png
  ```

**Requirements:**

- JSON must include all metrics
- Confusion matrix must be plotted using matplotlib

## TASK 3 — Add logging integration

Inside `eval.py` or a util:

Agent must add a logger that:

- Logs to file: `outputs/logs/eval_{timestamp}.log`
- Logs:
  - Model path used
  - Device used
  - Dataset size
  - Total accuracy
  - Per-class accuracy
  - Throughput
  - Model inference time

**No stdout spam — clean logs only.**

## TASK 4 — Throughput + Latency measurements

Agent must implement timing like:

```python
start = time.perf_counter()
outputs = model(images)
end = time.perf_counter()
batch_latency = end - start
```

**Compute:**

- Average inference time per sample
- Average batch latency
- `throughput = num_images_processed / total_time`

## TASK 5 — Confusion Matrix Plotting

Agent must:

- Compute confusion matrix using `sklearn.metrics.confusion_matrix`
- Plot with:
  - Color map
  - Axis labels (class names from dataset)
  - Values inside cells (optional but preferred)
- Save as PNG to metrics folder

## TASK 6 — Integrate with CLI

In `main.py`, add option:

```python
if cfg.mode == "eval":
    from scripts.run_eval import main_eval
    main_eval()
```

**Agent must NOT break existing training pipeline.**

## TASK 7 — Code Requirements

Agent must ensure:

- Functions are reusable
- No hardcoded paths
- All paths pulled from `config/base.yaml`
- Strict typing hints
- Consistent naming
- `torch.no_grad()` used in evaluation
- Batch size used exactly as defined in config
