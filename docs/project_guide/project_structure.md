# Project Structure

This document outlines the complete project structure (files and folders) for ImageNet Classification with Deep CNN.

## Root Directory Structure

```txt
imageNet_classification_with_deep_CNN/
├── .env                        # Environment variables (local)
├── .gitignore                  # Git ignore file
├── .python-version             # Python version specification
├── GEMINI.md                   # Project documentation
├── README.md                   # Project README
├── main.py                     # Main entry point
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── configs/                    # Configuration files
├── data/                       # Data directory
├── docs/                       # Documentation
├── models/                     # Model files and checkpoints
├── outputs/                    # Output files (logs, metrics, predictions)
├── scripts/                    # Utility scripts
└── src/                        # Source code
```

## Detailed Structure

### `/configs/` - Configuration Files

```txt
configs/
├── .gitkeep                    # Keeps directory in git
├── base.yaml                   # Base configuration
└── pytorch.yaml                # PyTorch experiment/training configuration
```

### `/data/` - Data Directory

```txt
data/
├── raw/                       # Raw data files
│   ├── .gitkeep
│   ├── translate.py           # Translation utility script
│   └── raw-img/               # Raw image files (Italian category names)
│       ├── cane/
│       ├── cavallo/
│       ├── elefante/
│       ├── farfalla/
│       ├── gallina/
│       ├── gatto/
│       ├── mucca/
│       ├── pecora/
│       ├── ragno/
│       └── scoiattolo/
└── processed/                 # Processed dataset (English category names)
    ├── .gitkeep
    ├── data_summary.json      # Dataset summary information
    ├── test/                  # Test split
    │   ├── butterfly/
    │   ├── cat/
    │   ├── chicken/
    │   ├── cow/
    │   ├── dog/
    │   ├── elephant/
    │   ├── horse/
    │   ├── sheep/
    │   ├── spider/
    │   └── squirrel/
    ├── train/                 # Training split
    │   ├── butterfly/
    │   ├── cat/
    │   ├── chicken/
    │   ├── cow/
    │   ├── dog/
    │   ├── elephant/
    │   ├── horse/
    │   ├── sheep/
    │   ├── spider/
    │   └── squirrel/
    └── val/                   # Validation split
        ├── butterfly/
        ├── cat/
        ├── chicken/
        ├── cow/
        ├── dog/
        ├── elephant/
        ├── horse/
        ├── sheep/
        ├── spider/
        └── squirrel/
```

### `/docs/` - Documentation

```txt
docs/
├── .gitkeep                    # Keeps directory in git
├── instructions/               # Setup and usage instructions
│   ├── general-setup-before-PyTorch.md
│   └── for_PyTorch/
│       ├── 01_PyTorch_core_implementation.md
│       └── 02_PyTorch_model_eval_and_metrics_logging.md
├── project_guide/              # Project guides
│   └── project_structure.md    # This file
├── project_progress/           # Project progress tracking
│   ├── phase-0-project-setup-&-data-pre-processing.md
│   └── progress.md
└── troubleshooting/            # Troubleshooting references
    └── cuda_issues.md
```

### `/models/` - Model Files

```txt
models/
├── .gitkeep                    # Keeps directory in git
└── pytorch/                    # Saved PyTorch model checkpoints
    ├── best_model.pt
    └── last_model.pt
```

### `/outputs/` - Output Files

```txt
outputs/
├── logs/                       # Training and execution logs
│   ├── .gitkeep
│   ├── download_data.log
│   ├── preprocess_data.log
│   ├── eval_20251203-002049.log
│   └── 2025*/train.log         # Multiple timestamped training logs
├── metrics/                    # Performance metrics
│   ├── .gitkeep
│   └── eval_20251203-002049/
│       ├── best_model_confusion_matrix.png
│       ├── best_model_metrics.json
│       ├── last_model_confusion_matrix.png
│       └── last_model_metrics.json
└── predictions/                # Model predictions
    └── .gitkeep
```

### `/scripts/` - Utility Scripts

```txt
scripts/
├── .gitkeep                   # Keeps directory in git
├── __init__.py                # Package marker
├── download_data.py           # Data downloading script
├── preprocess_data.py         # Data preprocessing script
└── pytorch/                   # PyTorch script entrypoints
    ├── __init__.py
    └── run_eval.py
```

### `/src/` - Source Code

```txt
src/
├── common/                    # Common utilities
│   ├── .gitkeep
│   ├── paths.py               # Path configurations
│   └── utils.py               # Utility functions
├── experiments/               # Experiment code
│   └── .gitkeep
├── pytorch/                   # PyTorch training/eval pipeline
│   ├── __pycache__/           # Bytecode cache
│   ├── data.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   └── transforms.py
└── tests/                     # Test files
    └── .gitkeep
```

## Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `configs/` | Configuration files for experiments and model parameters |
| `data/` | Storage for datasets (ImageNet data with raw and processed splits) |
| `docs/` | All project documentation and guides |
| `models/` | Trained model checkpoints and model definitions |
| `outputs/` | Results, logs, metrics, and predictions from experiments |
| `scripts/` | Utility scripts for data preparation and common tasks |
| `src/` | Main source code for the project |

## Notes

- `.gitkeep` files are used to maintain empty directories in version control.
- The `.venv` directory is excluded from commits (local virtual environment).
- The `source_material` directory contains reference papers/assignments.
- The `data/` directory contains both raw images (Italian labels) and processed splits (English labels).
- Processed data is organized into train/test/validation splits by class.
- All other directories and files are included in this structure documentation.
