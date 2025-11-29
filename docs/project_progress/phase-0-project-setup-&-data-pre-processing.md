# Phase 0: Project Setup & Data Pre-processing

## 1. Project Initialization
- **Dependency Management**: Switched to `uv` for faster and more reliable dependency management.
- **Configuration**: configured `pyproject.toml` with `hatchling` build system and `src` layout.
- **Environment**: Set up `.env` for secure credential management (Kaggle API).

## 2. Configuration System
- Created `configs/base.yaml` to centralize hyperparameters and paths.
- **Key Settings**:
  - Dataset: `alessiocorrado99/animals10`
  - Image Size: 128x128
  - Split Ratios: 80% Train, 10% Val, 10% Test
  - Seed: 42 (for reproducibility)

## 3. Common Utilities
- Implemented `src/common/paths.py` for absolute path management.
- Implemented `src/common/utils.py` for logging setup and deterministic seeding.

## 4. Dataset Handling
- **Download Script** (`scripts/download_data.py`):
  - robustly handles Kaggle authentication via `.env`.
  - Downloads and unzips the `alessiocorrado99/animals10` dataset.
- **Preprocessing Script** (`scripts/preprocess_data.py`):
  - **Translation**: Automatically translates Italian class names (e.g., `cane` -> `dog`, `farfalla` -> `butterfly`) to English.
  - **Resizing**: Resizes images to 128x128 pixels (LANCZOS resampling).
  - **Splitting**: Stratified split into Train, Validation, and Test sets.
  - **Summary**: Generates `data/processed/data_summary.json` with class counts.

## 5. Current Status
- **Raw Data**: Located in `data/raw/raw-img`.
- **Processed Data**: Located in `data/processed/{train,val,test}`.
- **Verification**: Verified folder structure and English class names.

## Next Steps
- Implement PyTorch Baseline Model (Task 2).
