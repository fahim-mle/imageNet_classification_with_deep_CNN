# Project Folder Structure

This document outlines the complete folder structure of the ImageNet Classification with Deep CNN project.

## Root Directory Structure

```txt
imageNet_classification_with_deep_CNN/
├── .gitignore                  # Git ignore file
├── .python-version             # Python version specification
├── GEMINI.md                   # Project documentation
├── main.py                     # Main entry point
├── pyproject.toml              # Python project configuration
├── README.md                   # Project README
├── uv.lock                     # Dependency lock file
├── configs/                    # Configuration files
├── data/                       # Data directory
├── docs/                       # Documentation
├── models/                     # Model files
├── outputs/                    # Output files
├── scripts/                    # Utility scripts
├── src/                        # Source code
```

## Detailed Structure

### `/configs/` - Configuration Files

```txt
configs/
├── .gitkeep                    # Keeps directory in git
└── base.yaml                   # Base configuration
```

### `/data/` - Data Directory

```txt
data/
├── raw/                       # Raw data files
│   ├── .gitkeep
│   ├── translate.py           # Translation utility script
│   └── raw-img/               # Raw image files
│       ├── cane/              # Italian category names
│       ├── cavallo/
│       ├── elefante/
│       ├── farfalla/
│       ├── gallina/
│       ├── gatto/
│       ├── mucca/
│       ├── pecora/
│       ├── ragno/
│       └── scoiattolo/
└── processed/                 # Processed dataset
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
│   └── general-setup-before-PyTorch.md
├── project_guide/              # Project guides
│   └── folder_structure.md     # This file
└── project_progress/           # Project progress tracking
    └── phase-0-project-setup-&-data-pre-processing.md
```

### `/models/` - Model Files

```txt
models/
└── .gitkeep                    # Keeps directory in git
```

### `/outputs/` - Output Files

```txt
outputs/
├── logs/                       # Training and execution logs
│   └── .gitkeep
├── metrics/                    # Performance metrics
│   └── .gitkeep
└── predictions/                # Model predictions
    └── .gitkeep
```

### `/scripts/` - Utility Scripts

```txt
scripts/
├── .gitkeep                   # Keeps directory in git
├── download_data.py           # Data downloading script
└── preprocess_data.py         # Data preprocessing script
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

- `.gitkeep` files are used to maintain empty directories in version control
- The `.venv` directory is excluded as it contains virtual environment files
- The `source_material` directory is excluded as per project requirements
- The `data/` directory contains both raw images (with Italian category names) and processed dataset splits
- Processed data is organized into train/test/validation splits with English category names
- All other directories and files are included in this structure documentation
