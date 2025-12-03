import tensorflow as tf

def create_model(cfg):
    """
    Create TensorFlow/Keras model based on config.
    """
    architecture = cfg['model']['architecture']
    image_size = tuple(cfg['dataset']['image_size'])
    num_classes = cfg['model']['num_classes']
    pretrained = cfg['model']['pretrained']
    mixed_precision = cfg['training'].get('mixed_precision', False)

    # Set mixed precision policy
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    input_shape = (*image_size, 3)
    weights = "imagenet" if pretrained else None

    # Load backbone
    if architecture == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            weights=weights,
            pooling="avg"
        )
    elif architecture == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=input_shape,
            weights=weights,
            pooling="avg"
        )
    elif architecture == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=input_shape,
            weights=weights,
            pooling="avg"
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Custom head
    x = tf.keras.layers.Dense(256, activation="relu")(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output layer
    # For mixed precision, output should be float32 for stability
    activation = "softmax"
    if mixed_precision:
        activation = "linear" # We'll apply softmax separately or use from_logits=True in loss
        # Actually, standard practice is to use float32 dtype for output layer
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    return model
