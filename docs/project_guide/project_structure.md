# Project Structure

Current layout of ImageNet Classification with Deep CNN (PyTorch and TensorFlow pipelines).

## Top-level layout

```txt
imageNet_classification_with_deep_CNN/
├── .env                  # Local environment variables
├── .gitignore
├── .python-version       # Python version pin
├── GEMINI.md             # Project notes
├── README.md             # Project overview and usage
├── main.py               # Entrypoint/CLI hook
├── pyproject.toml        # Project config and dependencies
├── uv.lock               # Locked dependency set
├── configs/              # Experiment/training configs
├── data/                 # Raw and processed datasets
├── docs/                 # Guides and progress logs
├── models/               # Saved checkpoints
├── outputs/              # Logs, metrics, figures, predictions
├── scripts/              # Runner and helper scripts
├── source_material/      # Reference papers/assignments
├── src/                  # Source code packages
└── .venv/                # Local virtual environment (untracked)
```

## Configurations

```txt
configs/
├── base.yaml             # Shared/default settings
├── pytorch.yaml          # PyTorch experiment configuration
└── tensorflow.yaml       # TensorFlow experiment configuration
```

## Data layout

```txt
data/
├── raw/                  # Original dataset (Italian class names)
│   ├── translate.py      # Utility to translate labels
│   └── raw-img/
│       ├── cane/ cavallo/ elefante/ farfalla/ gallina/
│       ├── gatto/ mucca/ pecora/ ragno/ scoiattolo/
├── processed/            # Cleaned dataset (English class names)
│   ├── data_summary.json # Dataset stats
│   ├── train/            # Training split (per-class folders)
│   ├── val/              # Validation split (per-class folders)
│   └── test/             # Test split (per-class folders)
```

## Documentation

```txt
docs/
├── instructions/
│   ├── evaluation/
│   │   └── 01_instruction_for_evaluating_trained_models.md
│   ├── for_PyTorch/
│   │   ├── general-setup-before-PyTorch.md
│   │   ├── 01_PyTorch_core_implementation.md
│   │   └── 02_PyTorch_model_eval_and_metrics_logging.md
│   └── for_tensorflow/
│       ├── 01_setting_up_tensorflow_config.md
│       ├── 01_TF_core_implementation.md
│       ├── 02_TF_core_full_implementation.md
│       └── 02_TF_training_and_model_build.md
├── project_guide/
│   └── project_structure.md   # This file
├── project_progress/
│   ├── phase-0-project-setup-&-data-pre-processing.md
│   └── progress.md
└── troubleshooting/
    └── cuda_issues.md
```

## Scripts

```txt
scripts/
├── download_data.py           # Pulls raw dataset
├── preprocess_data.py         # Cleans/splits data
├── evaluate_all_models.py     # Batch evaluation helper
├── pytorch/
│   └── run_eval.py            # PyTorch evaluation entrypoint
└── tensorflow/
    ├── run.sh                 # Convenience runner
    ├── run_train.py           # TensorFlow training
    ├── run_eval.py            # TensorFlow evaluation
    ├── verify_data.py         # Dataset checks
    ├── verify_model.py        # Model sanity checks
    └── verify_eval.py         # Evaluation sanity checks
```

## Source code

```txt
src/
├── common/
│   ├── paths.py               # Centralized path helpers
│   └── utils.py               # Shared utilities
├── experiments/               # Placeholder for experiment scripts
├── pytorch/                   # PyTorch pipeline
│   ├── data.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   └── transforms.py
├── tensorflow/                # TensorFlow pipeline
│   ├── data.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
└── tests/                     # Test scaffolding (currently placeholder)
```

## Models

```txt
models/
├── pytorch/
│   ├── best_model.pt
│   └── last_model.pt
└── tf/
    ├── best_model.keras
    ├── last_model.keras
    └── test_model.keras
```

## Outputs and artifacts

```txt
outputs/
├── figures/                       # Visualizations per evaluation
│   └── eval_<timestamp>/
│       ├── confusion_matrix_best_model_pytorch.png
│       ├── confusion_matrix_best_model_tensorflow.png
│       ├── per_class_accuracy_best_model_[pytorch|tensorflow].png
│       ├── roc_curve_best_model_[pytorch|tensorflow].png
│       └── sample_predictions_best_model_[pytorch|tensorflow].png
├── logs/                          # Execution + training logs
│   ├── download_data.log
│   ├── preprocess_data.log
│   ├── eval_<timestamp>.log
│   └── <timestamp>/train.log      # Training sessions (timestamped folders)
├── metrics/                       # Aggregated metrics
│   ├── eval_<timestamp>/          # PyTorch + TensorFlow eval outputs
│   │   ├── best_model_pytorch_[train|val|test]_metrics.json
│   │   └── best_model_tensorflow_[train|val|test]_metrics.json
│   └── tf/<timestamp>/            # TensorFlow-only eval artifacts
│       ├── confusion_matrix.png
│       └── metrics.json
├── predictions/
│   └── predictions_best_model_[pytorch|tensorflow]_<timestamp>.csv
└── tf_logs/                       # TensorFlow training/eval logs (CSV/log)
    ├── eval_<timestamp>.log
    └── train_<timestamp>.csv
```

## Reference material

```txt
source_material/
├── 1_Imagenet-classification-with-deep-convolutional-neural-networks.pdf
└── MA5852 Assignment 3 Brisbane.pdf
```

## Notes

- `.gitkeep` files keep otherwise-empty directories in version control.
- Timestamped run folders follow the pattern `<YYYYMMDD-HHMMSS>`.
- `.venv/` is local-only and should not be committed.
- Raw images use Italian labels; processed splits use English labels across train/val/test.
