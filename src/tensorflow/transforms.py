import tensorflow as tf

def get_image_transforms(cfg):
    """
    Return a preprocessing function usable inside tf.data
    """
    image_size = tuple(cfg['dataset']['image_size'])

    # Augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
    ])

    def preprocess(image, label, training=False):
        # Resize and rescale
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0

        if training:
            image = data_augmentation(image, training=True)

        return image, label

    return preprocess
