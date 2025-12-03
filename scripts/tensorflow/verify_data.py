import os
import sys
import tensorflow as tf
from src.common.utils import load_config
from src.tensorflow.data import load_dataset, prepare_tf_dataset

def verify_data_pipeline():
    cfg = load_config("configs/tensorflow.yaml")

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_dataset(cfg)

    print(f"Class names: {class_names}")

    if train_ds:
        print("Preparing train dataset...")
        train_ds = prepare_tf_dataset(train_ds, cfg, training=True)
        for images, labels in train_ds.take(1):
            print(f"Train batch shape: {images.shape}")
            print(f"Train labels shape: {labels.shape}")
            print(f"Image range: [{tf.reduce_min(images)}, {tf.reduce_max(images)}]")

    if val_ds:
        print("Preparing val dataset...")
        val_ds = prepare_tf_dataset(val_ds, cfg, training=False)
        for images, labels in val_ds.take(1):
            print(f"Val batch shape: {images.shape}")

if __name__ == "__main__":
    verify_data_pipeline()
