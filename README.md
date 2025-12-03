# ImageNet Classification with Deep CNNs

PyTorch and TensorFlow pipelines for the Animals-10 dataset with reproducible training, evaluation, and reporting. Outputs (logs, metrics, figures, predictions) for recent runs are stored under `outputs/` and checkpoints under `models/`.

## Quickstart

```bash
# 1) Install uv (recommended) https://docs.astral.sh/uv/getting-started/
# 2) Sync environment (Python >= 3.12)
uv sync

# 3) Set Kaggle credentials for dataset download
echo "KAGGLE_USERNAME=<your_username>" >> .env
echo "KAGGLE_KEY=<your_key>" >> .env

# 4) Download and preprocess Animals-10
uv run python scripts/download_data.py
uv run python scripts/preprocess_data.py
```

Dataset summary after preprocessing (from `data/processed/data_summary.json`):
- Classes: chicken, spider, dog, cat, cow, sheep, butterfly, elephant, squirrel, horse
- Counts: train 20,933 | val 2,623 | test 2,623

## Train & Evaluate

### PyTorch (ResNet18, pretrained)
```bash
# Train
uv run python main.py --framework pytorch --train
# Eval (uses last/best in models/pytorch/)
uv run python main.py --framework pytorch --eval
```
Artifacts:
- Checkpoints: `models/pytorch/`
- Logs: `outputs/logs/<timestamp>/train.log` (plus `outputs/logs/eval_<timestamp>.log`)
- Metrics/figures/predictions: `outputs/metrics/eval_<timestamp>/`, `outputs/figures/eval_<timestamp>/`, `outputs/predictions/`

### TensorFlow (MobileNetV2, pretrained)
```bash
# Train
uv run python main.py --framework tensorflow --train
# Eval (defaults to models/tf/best_model.keras)
uv run python main.py --framework tensorflow --eval
# Convenience runner
./scripts/tensorflow/run.sh main.py --framework tensorflow --train
```
Artifacts:
- Checkpoints: `models/tf/`
- Logs: `outputs/tf_logs/train_<timestamp>.csv` and `outputs/tf_logs/eval_<timestamp>.log`
- Metrics/figures/predictions: `outputs/metrics/eval_<timestamp>/`, `outputs/figures/eval_<timestamp>/`, `outputs/predictions/`

### Evaluate all saved models together
```bash
uv run python scripts/evaluate_all_models.py
```
Generates metrics, figures (confusion matrices, ROC, per-class accuracy), and CSV predictions for both frameworks in a new `eval_<timestamp>` folder.

## Configuration
- Dataset defaults: `configs/base.yaml` (Animals-10 Kaggle source, 128x128 resize, 80/10/10 split, seed 42).
- PyTorch-specific: `configs/pytorch.yaml` (ResNet18, Adam, weight decay).
- TensorFlow-specific: `configs/tensorflow.yaml` (MobileNetV2, Adam, mixed precision toggle).
Adjust these to change architectures, learning rates, batch sizes, or splits.

## Results (latest captured in `outputs/metrics/eval_20251203-225445/`)
- TensorFlow best model: val acc 0.920, test acc 0.909 (`models/tf/best_model.keras`, metrics JSONs in the path above).
- PyTorch best model: val acc 0.858, test acc 0.846 (`models/pytorch/best_model.pt`).
- Visuals (confusion matrices, ROC, per-class accuracy, sample predictions): `outputs/figures/eval_20251203-225445/`.
- Prediction CSVs: `outputs/predictions/predictions_best_model_[pytorch|tensorflow]_20251203-225445.csv`.

## Troubleshooting
- Kaggle auth issues: ensure `.env` has `KAGGLE_USERNAME` and `KAGGLE_KEY`. Re-run `uv run python scripts/download_data.py`.
- No data found: confirm `data/raw/raw-img/` contains class folders; then rerun preprocessing.
- GPU use: PyTorch auto-selects CUDA/MPS if available; TensorFlow uses mixed precision when enabled in `configs/tensorflow.yaml`.
