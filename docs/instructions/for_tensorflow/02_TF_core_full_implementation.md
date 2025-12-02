# 🚀 TF-CORE PHASE 2 — FULL IMPLEMENTATION INSTRUCTION SET

This phase implements the actual TensorFlow training pipeline inside:

```
src/tensorflow/
```

Everything here builds on Phase 1 scaffolding.

## 🟦 TASK 1 — Implement TensorFlow Transforms

**File:** `src/tensorflow/transforms.py`

Implement:

### get_image_transforms(cfg)

Return a preprocessing function usable inside `tf.data`

**Requirements:**

- Decode JPG/PNG
- Convert to float32
- Resize to `cfg.dataset.image_size`
- Normalize pixel values to [0,1]

**Keep augmentation minimal for now:**

- `RandomFlip("horizontal")`
- `RandomRotation(0.05)`

> **Note:** Augmentation must apply only to training dataset

**Final output shape:** `(H, W, 3)`

---

## 🟦 TASK 2 — Implement Dataset Loader (tf.data)

**File:** `src/tensorflow/data.py`

Implement:

### load_dataset(cfg)

**Should:**

Build dataset paths from:

- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

Use `tf.keras.preprocessing.image_dataset_from_directory` with:

```python
image_size = cfg.dataset.image_size
label_mode = "int"
shuffle = True  # for train only
```

**Returns:** `train_ds, val_ds, test_ds, class_names`

### prepare_tf_dataset(ds, cfg, training=False)

Apply:

- Transforms via `.map()`
- Shuffle only for training
- `.batch(cfg.training.batch_size)`
- `.prefetch(tf.data.AUTOTUNE)`

### get_class_names(cfg)

Implement directory scan of `data/processed/train/`.

---

## 🟦 TASK 3 — Implement TensorFlow Model Builder

**File:** `src/tensorflow/model.py`

Implement:

### create_model(cfg)

**Rules:**

Load backbone from `keras.applications` based on `cfg.model.architecture`

**Example:**

```python
base = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(*cfg.dataset.image_size, 3),
    weights="imagenet" if cfg.model.pretrained else None,
    pooling="avg"
)
```

**Add custom head:**

```python
x = tf.keras.layers.Dense(256, activation="relu")(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(cfg.model.num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=outputs)
```

**If `cfg.training.mixed_precision == true`:**

```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

**Return compiled model?**
→ No, compilation is handled in `train.py`

---

## 🟦 TASK 4 — Implement Model Compilation

**File:** `src/tensorflow/train.py`

Add:

### compile_model(model, cfg)

Create optimizer based on cfg:

```python
opt = tf.keras.optimizers.Adam(
    learning_rate=cfg.training.learning_rate
)
```

- **Loss:** `"sparse_categorical_crossentropy"`
- **Metrics:** `["accuracy"]`

Return compiled model.

---

## 🟦 TASK 5 — Implement Training Loop

**File:** `src/tensorflow/train.py`

Implement:

### train_model(cfg)

**Steps:**

1. Load datasets using `load_dataset + prepare_tf_dataset`
2. Build model using `create_model(cfg)`
3. Compile model via `compile_model`
4. Set callbacks:

   **ModelCheckpoint:**

   ```python
   models/tf/best_model.h5
   ```

   **CSVLogger:**

   ```python
   outputs/tf_logs/train_{timestamp}.csv
   ```

   **EarlyStopping** (patience=3)

5. Call:

   ```python
   history = model.fit(
       train_ds,
       validation_data=val_ds,
       epochs=cfg.training.epochs,
   )
   ```

6. Save final model:

   ```python
   models/tf/last_model.h5
   ```

7. Log metrics (accuracy & loss per epoch)

---

## 🟦 TASK 6 — Implement Evaluation

**File:** `src/tensorflow/eval.py`

Implement:

### evaluate_model(cfg, model_path)

**Steps:**

1. Load model:

   ```python
   model = tf.keras.models.load_model(model_path)
   ```

2. Load test dataset

3. Compute:
   - Overall accuracy
   - Per-class accuracy
   - Confusion matrix
   - Inference latency per batch
   - Throughput

4. Save metrics to:

   ```
   outputs/metrics/tf/{timestamp}/
   ```

**Files saved:**

- `metrics.json`
- `confusion_matrix.png`

5. Log everything to:

   ```
   outputs/tf_logs/eval_{timestamp}.log
   ```

---

## 🟦 TASK 7 — Add TF CLI Entrypoints

Add folder:

```
scripts/tensorflow/
    run_train.py
    run_eval.py
```

### run_train.py

```python
load config (tensorflow.yaml)
run train_model(cfg)
```

### run_eval.py

```python
load config
run evaluate_model(cfg, model_path)
```

---

## 🟦 TASK 8 — Documentation

Create: `docs/instructions/for_TensorFlow/02_TF_training_and_model_build.md`

Document:

- How dataset pipeline works
- How to train
- How to evaluate
- Where checkpoints/logs go
- Expected outputs

---

## 🟦 TASK 9 — Project Isolation Rules

Your agent **MUST:**

- Keep TF code fully inside `src/tensorflow/` and `scripts/tensorflow/`
- Write model files ONLY under `models/tf/`
- Write logs ONLY under `outputs/tf_logs/`
- Write metrics ONLY under `outputs/metrics/tf/`
- Never import PyTorch modules
- Never touch PyTorch configs

---

## 🟦 TASK 10 — Outputs of Phase 2

When complete, your agent will have created:

- ✅ Full working TensorFlow training pipeline
- ✅ Full dataset pipeline
- ✅ Full model-building logic
- ✅ Full compile logic
- ✅ Full training loop
- ✅ Full evaluation logic
- ✅ CLI runners
- ✅ Logs, metrics, and saved models

And nothing will conflict with your PyTorch system.
