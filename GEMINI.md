# Gemini Code Generator Guidelines

## Purpose

Gemini acts as the main code generator for the PyTorch workflow. It must produce clean, modular, readable code following the project structure.

## Behavior Rules (Gemini MUST follow these)

### 1. Code Style

- Write Python 3.10+ compatible code
- Keep functions small and modular
- Each file should have one clear responsibility
- Use typing hints everywhere
- Add docstrings but no unnecessary comments
- Raise meaningful exceptions instead of silent fails

### 2. Folder + File Rules

Gemini MUST align code with this structure:

```
src/
    pytorch/
        __init__.py
        data.py
        transforms.py
        model.py
        train.py
        eval.py
        utils.py
configs/
    base.yaml
    pytorch.yaml
```

- No throwing files randomly anywhere
- If a file doesn't exist, Gemini creates it

### 3. Config Handling

Gemini MUST use base.yaml + pytorch.yaml merged (pytorch overrides base).

Gemini MUST load configs via a clean helper:

```python
from src.common.utils import load_config
cfg = load_config("configs/pytorch.yaml")
```

- Values in dataset: and training: MUST flow from the config, not hardcoded

### 4. Path Handling

Gemini MUST import paths via:

```python
from src.common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
```

- No hardcoded path strings in the code

### 5. Logging Rules

- Use Python's logging module
- Logs go to: outputs/logs/
- Each run gets its own timestamped folder
- Gemini MUST NOT print random debugging noise

### 6. Device Management

- Automatically detect cuda → mps → cpu
- Code must support multi-device seamlessly

### 7. Model Saving Rules

Save checkpoints to:

```
models/pytorch/
    best_model.pt
    last_model.pt
```

- No weird filenames

### 8. Imports

Prefer:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
```

- NOT wildcard imports

### 9. Error Handling

Gemini must:

- Validate folder existence
- Validate dataset structure
- Validate config fields
- Raise useful errors when missing files/keys

### 10. Output

- Gemini's responses MUST return only the code or file content requested
- No fluff, no explanations unless explicitly asked
