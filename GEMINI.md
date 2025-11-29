# 🤖 ImageNet Classification with Deep CNN

> 🚀 **Building and comparing deep convolutional neural network models for image classification across multiple modern ML frameworks**

---

## 📋 Project Overview

This project focuses on building and comparing deep convolutional neural network (CNN) models for image classification using multiple modern machine-learning frameworks. Inspired by the foundational AlexNet architecture, the goal is to design, train, and evaluate custom CNNs while also leveraging state-of-the-art pretrained models.

### 🎯 Objectives

- 🔧 Design custom CNN architectures based on AlexNet
- 🏗️ Implement models across three major frameworks
- 📊 Compare performance and workflows
- 🔬 Analyze architectural impacts on results
- 📈 Establish reproducible experimentation foundation

### 🛠️ Technology Stack

The project includes **three parallel implementations**:

| Framework | Type | Purpose |
|-----------|------|---------|
| 🔥 **PyTorch** | Custom baseline | Deep learning research framework |
| 🧠 **TensorFlow/Keras** | Custom baseline | Production-ready framework |
| 🤗 **Hugging Face** | Transfer learning | State-of-the-art pretrained models |

All models will be trained and tested on a chosen Kaggle image dataset, using consistent preprocessing, evaluation metrics, and output logging. The final objective is to analyze performance differences, understand architectural impacts, and produce a clear comparison of deep-learning workflows across different frameworks.

This repo serves as a clean, maintainable foundation for experimentation, reproducibility, and extension into broader computer-vision research or MLOps practices.

---

## 🏗️ Building and Running

### 1️⃣ Project Build Rules

📁 **Code Organization**
- ✅ Keep all code inside `src/`, organized by purpose (`common/`, `experiments/`, etc.)
- 📦 Add new dependencies only in `requirements.txt`
- 🚫 Never hard-code local paths
- ⚙️ Use config files (YAML/JSON under `configs/`) for hyperparameters, dataset paths, model settings

🔧 **Code Generation Guidelines**
- 📝 Follow PEP8 formatting
- 📦 Keep files small and modular
- 🔧 Put reusable utilities into `src/common/`
- 🚫 Never store datasets or checkpoints in Git (only `.gitkeep` files)

### 2️⃣ How to Run the Project

For every model (PyTorch, TF, HF) the agent must:

📋 **Setup Steps**
1. ⚙️ Load configs from `configs/`
2. 📂 Load dataset from `data/processed/` (document expected folder structure)
3. 🏃 Run training using scripts under `scripts/`:
   - `scripts/train_pytorch.py`
   - `scripts/train_tensorflow.py`
   - `scripts/train_hf.py`

📊 **Output Management**
Save all outputs into the correct subfolders:
- 📊 **logs** → `outputs/logs/`
- 📈 **metrics** → `outputs/metrics/`
- 🔮 **predictions** → `outputs/predictions/`
- 💾 **model weights** → `models/`

🔄 **Runtime Requirements**
- 📈 Always log: loss curves, accuracy curves, confusion matrix (if classification)
- 💾 Always save: final model, config used, training summary file (JSON)

### 3️⃣ How to Test the Project

The agent must implement **three layers of testing**:

#### 🧪 A. Sanity Tests (required for every training script)
- ✅ Check dataset loads without errors
- ✅ Check model instantiates correctly
- ✅ Run a single forward pass on 1–2 samples
- ✅ Verify training loop for 1 batch works

#### 🔧 B. Functional Tests (placed in `src/tests/`)
- 🔄 Test preprocessing pipeline
- 💾 Test model saves/loads correctly
- 📊 Test metrics computation (accuracy, loss)

#### 📈 C. Evaluation Tests
After training, always produce:
- 📊 Accuracy on train/test split
- 📈 Loss curves
- 📄 Metrics JSON file
- 🖼️ 5–10 example predictions saved as images or text

### 4️⃣ Execution Workflow for the Agent

Whenever the agent adds or modifies code, it must follow this sequence:

1. 📦 Update dependencies if needed → `requirements.txt`
2. 📝 Generate code inside the correct folder
3. 🧪 Run sanity tests
4. 📖 Document usage in README or a small HOW-TO comment at top of script
5. 💾 Commit changes with a meaningful message

### 5️⃣ Branching Workflow

🌳 **Branch Strategy**
- 🌟 **Main branch** → stable scaffold and docs
- 🚀 **New model** → create feature branch (`feature/pytorch_baseline`, etc.)

🔄 **Merge Process**
After finishing:
- ✅ Ensure code runs end-to-end with sanity tests
- 🔄 Merge back into main

### 6️⃣ Efficiency & Optimization Rules

⚡ **Performance Guidelines**
- 🎯 Prefer small batches and lower-res images for development (due to 4 GB VRAM)
- 🎲 Use deterministic seeding
- 🔄 Use shared data loader utilities for all frameworks
- 💾 Cache preprocessed data in `data/processed/`

---

## 👨‍💻 Development Conventions

### 📝 Coding Standards
- 🐍 Follow PEP8 formatting for Python code
- 📝 Use clear, descriptive variable names
- 📦 Keep functions small and focused
- 📖 Add docstrings to all public functions and classes

### 🧪 Testing Practices
- ✅ Write tests for all new functionality
- 🔄 Run tests before committing changes
- 📊 Maintain test coverage above 80%
- 🧪 Include both unit and integration tests

### 📚 Documentation
- 📖 Keep README files up to date
- 💬 Comment complex logic
- 📊 Document API changes
- 🔄 Update configuration examples

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### ⚙️ Installation
```bash
# Clone the repository
git clone <repository-url>
cd imageNet_classification_with_deep_CNN

# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p data/{raw,processed} outputs/{logs,metrics,predictions} models configs
```

### 🏃 Quick Start
```bash
# Train PyTorch model
python scripts/train_pytorch.py --config configs/pytorch_config.yaml

# Train TensorFlow model
python scripts/train_tensorflow.py --config configs/tensorflow_config.yaml

# Train Hugging Face model
python scripts/train_hf.py --config configs/hf_config.yaml
```

---

## 📊 Project Structure

```
imageNet_classification_with_deep_CNN/
├── 📁 src/                    # Source code
│   ├── 📁 common/             # Shared utilities
│   ├── 📁 experiments/        # Experiment scripts
│   └── 📁 tests/              # Test files
├── 📁 scripts/               # Training scripts
├── 📁 configs/               # Configuration files
├── 📁 data/                  # Data directories
│   ├── 📁 raw/               # Raw datasets
│   └── 📁 processed/         # Processed datasets
├── 📁 outputs/               # Training outputs
│   ├── 📁 logs/              # Training logs
│   ├── 📁 metrics/           # Evaluation metrics
│   └── 📁 predictions/       # Model predictions
├── 📁 models/                # Saved model weights
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md              # Project documentation
└── 📄 GEMINI.md              # This file
```
---

*Last updated: November 2025*
