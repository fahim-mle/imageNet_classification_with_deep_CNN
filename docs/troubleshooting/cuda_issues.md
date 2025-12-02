# Hardware Acceleration & CUDA Issues

## Overview
This document outlines the issues encountered when attempting to enable GPU acceleration for the PyTorch implementation.

## Current Status
- **PyTorch Version**: `2.5.1+cu121` (Installed via `uv` with CUDA 12.1 support)
- **CUDA Available**: `False`
- **Device Used**: CPU

## Findings

### 1. Missing NVIDIA Drivers in Environment
The command `nvidia-smi`, which is used to monitor NVIDIA GPU status and driver version, is **not found** in the current shell environment.

```bash
$ nvidia-smi
Command 'nvidia-smi' not found
```

This indicates that the NVIDIA drivers are either not installed or not exposed to the current environment (e.g., running inside a container without `--gpus all` or equivalent).

### 2. PyTorch Cannot Detect CUDA
Despite installing the correct CUDA-enabled version of PyTorch, the library cannot see the GPU.

```python
import torch
print(torch.cuda.is_available())
# Output: False
```

### 3. Dependency Configuration is Correct
The `pyproject.toml` was successfully updated to point to the PyTorch CUDA index:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

And `uv sync` successfully installed the `+cu121` binaries.

## Conclusion
The project is correctly configured for GPU usage.

**Update**: As of 2025-12-02, the user has installed NVIDIA drivers (v535).
- `nvidia-smi` is functional.
- PyTorch successfully detects the GPU (`cuda:0`).
- Training runs on the GPU with ~5x speedup compared to CPU.

**Resolution**: The issue was missing drivers in the environment, which has been resolved.
