# General Setup Before PyTorch

## Task List

### 1. Dataset Handling

- [ ] Implement Kaggle API download script for Animals-10
- [ ] Create folder structure under `data/raw` and `data/processed`
- [ ] Write a preprocessing script:
  - Unzip + organize images
  - Resize (128–160px recommended)
  - Split train/val/test
  - Save metadata (csv or json)

### 2. Configuration System

- [ ] Create global config file (`configs/base.yaml`)
- [ ] Include:
  - Dataset paths
  - Image size
  - Batch size defaults
  - Normalization stats (placeholder)

### 3. Common Utils

- [ ] `src/common/logger.py` → pretty logs
- [ ] `src/common/paths.py` → centralize all paths
- [ ] `src/common/seed.py` → reproducibility helper
- [ ] `src/common/data_utils.py` → reusable helpers
- [ ] Maybe simple file-check helpers

### 4. Basic Test/Sanity Scripts

- [ ] Verify dataset loads
- [ ] Verify processed images
- [ ] Verify splits
- [ ] Ensure nothing breaks before doing ML

### 5. Documentation

- [ ] Update README placeholders: dataset + setup instructions
- [ ] Update your coding-agent instruction file if needed
