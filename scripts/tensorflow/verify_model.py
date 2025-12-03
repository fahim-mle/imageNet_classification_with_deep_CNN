import os
import tensorflow as tf
from src.common.utils import load_config
from src.tensorflow.model import create_model
from src.tensorflow.train import compile_model

def verify_model_training():
    cfg = load_config("configs/tensorflow.yaml")

    print("Creating model...")
    try:
        model = create_model(cfg)
        model.summary()
        print("Model created successfully.")
    except Exception as e:
        print(f"Failed to create model: {e}")
        return

    print("Compiling model...")
    try:
        model = compile_model(model, cfg)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Failed to compile model: {e}")
        return

    print("Verification complete.")

if __name__ == "__main__":
    verify_model_training()
