import tensorflow as tf
import os
from src.common.paths import DATA_DIR
from src.tensorflow.transforms import get_image_transforms

def load_dataset(cfg):
    """
    Load train/val/test folders from data/processed/
    Return train_ds, val_ds, test_ds, class_names
    """
    batch_size = cfg['training']['batch_size']
    image_size = tuple(cfg['dataset']['image_size'])

    train_dir = os.path.join(DATA_DIR, "processed", "train")
    val_dir = os.path.join(DATA_DIR, "processed", "val")
    test_dir = os.path.join(DATA_DIR, "processed", "test")

    # Helper to load dataset
    def load_from_dir(directory, shuffle=False):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found.")
            return None

        return tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=None,
            color_mode="rgb",
            batch_size=None, # We batch later
            image_size=image_size,
            shuffle=shuffle,
            seed=42 if shuffle else None,
        )

    train_ds = load_from_dir(train_dir, shuffle=True)
    val_ds = load_from_dir(val_dir, shuffle=False)
    test_ds = load_from_dir(test_dir, shuffle=False)

    class_names = get_class_names(cfg)

    return train_ds, val_ds, test_ds, class_names

def prepare_tf_dataset(ds, cfg, training=False):
    """
    Apply transforms, batching, shuffling, and prefetching
    """
    if ds is None:
        return None

    batch_size = cfg['training']['batch_size']
    shuffle_buffer = cfg['dataset']['shuffle_buffer']

    num_parallel_calls = cfg['dataset'].get('num_parallel_calls', 'auto')
    if num_parallel_calls == 'auto':
        autotune = tf.data.AUTOTUNE
    else:
        autotune = int(num_parallel_calls)

    preprocess_fn = get_image_transforms(cfg)

    # Wrap preprocess to pass training flag
    def _preprocess(image, label):
        return preprocess_fn(image, label, training=training)

    ds = ds.map(_preprocess, num_parallel_calls=autotune)

    if training:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune)

    return ds

def get_class_names(cfg):
    """
    Scan data/processed/train/ and return folder names
    """
    train_dir = os.path.join(DATA_DIR, "processed", "train")
    if not os.path.exists(train_dir):
        return []

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return class_names
