import os
import tensorflow as tf
from src.common.utils import load_config
from src.common.paths import MODELS_DIR
from src.tensorflow.model import create_model
from src.tensorflow.eval import evaluate_model

def verify_evaluation():
    cfg = load_config("configs/tensorflow.yaml")

    # Create dummy model
    print("Creating dummy model...")
    model = create_model(cfg)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Save dummy model
    checkpoint_dir = os.path.join(MODELS_DIR, "tf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "test_model.keras")
    model.save(model_path)
    print(f"Saved dummy model to {model_path}")

    # Run evaluation
    print("Running evaluation...")
    evaluate_model(cfg, model_path)
    print("Verification complete.")

if __name__ == "__main__":
    verify_evaluation()
