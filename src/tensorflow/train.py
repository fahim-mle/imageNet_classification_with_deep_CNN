import tensorflow as tf
import os
from datetime import datetime
from src.common.paths import OUTPUTS_DIR, MODELS_DIR
from src.tensorflow.data import load_dataset, prepare_tf_dataset
from src.tensorflow.model import create_model
from src.tensorflow.utils import setup_logging

def compile_model(model, cfg):
    """
    Compile model with optimizer, loss, and metrics
    """
    learning_rate = cfg['training']['learning_rate']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def train_model(cfg):
    """
    Execute training pipeline
    """
    logger = setup_logging(cfg)
    logger.info("Starting TensorFlow training...")

    # 1. Load Data
    logger.info("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_dataset(cfg)

    if train_ds is None:
        logger.error("Failed to load datasets.")
        return

    train_ds = prepare_tf_dataset(train_ds, cfg, training=True)
    val_ds = prepare_tf_dataset(val_ds, cfg, training=False)

    # 2. Create Model
    logger.info(f"Creating model: {cfg['model']['architecture']}")
    model = create_model(cfg)

    # 3. Compile Model
    logger.info("Compiling model...")
    model = compile_model(model, cfg)

    # 4. Setup Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Checkpoint
    checkpoint_dir = os.path.join(MODELS_DIR, "tf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.keras") # .keras is preferred format now

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )

    # CSV Logger
    tf_logs_dir = os.path.join(OUTPUTS_DIR, "tf_logs")
    os.makedirs(tf_logs_dir, exist_ok=True)
    csv_log_path = os.path.join(tf_logs_dir, f"train_{timestamp}.csv")

    csv_logger_cb = tf.keras.callbacks.CSVLogger(csv_log_path)

    # Early Stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    callbacks = [checkpoint_cb, csv_logger_cb, early_stopping_cb]

    # 5. Train
    epochs = cfg['training']['epochs']
    logger.info(f"Starting training for {epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 6. Save Final Model
    last_model_path = os.path.join(checkpoint_dir, "last_model.keras")
    model.save(last_model_path)
    logger.info(f"Saved last model to {last_model_path}")

    logger.info("Training complete.")
