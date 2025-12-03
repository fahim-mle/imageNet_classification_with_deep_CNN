import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from src.common.paths import OUTPUTS_DIR, MODELS_DIR
from src.pytorch.eval import evaluate
from src.pytorch.data import get_dataloaders
from src.pytorch.model import create_model

def train_model(cfg, model, loaders):
    """
    Train a PyTorch model using the provided configuration and data loaders.
    
    Trains `model` for the configured number of epochs, evaluates on the validation loader each epoch, logs progress to a timestamped outputs log, and saves both the latest and best model state_dicts to the project models directory. The function selects CUDA if available, then MPS, otherwise CPU.
    
    Parameters:
        cfg (dict): Configuration dictionary. Required keys:
            - 'optimizer.name' (str): Optimizer identifier, e.g. 'adam' (otherwise SGD is used).
            - 'optimizer.weight_decay' (numeric): Weight decay for the optimizer.
            - 'training.learning_rate' (numeric): Learning rate.
            - 'training.epochs' (int): Number of training epochs.
            - 'loss.name' (str): Loss identifier, e.g. 'cross_entropy'.
        model: A PyTorch nn.Module instance to be trained; its parameters will be moved to the selected device.
        loaders (tuple): (train_loader, val_loader, test_loader). Only the train and val loaders are used.
    
    Side effects:
        - Writes training logs to OUTPUTS_DIR/logs/<timestamp>/train.log and streams to console.
        - Saves model state_dict to MODELS_DIR/pytorch/last_model.pt each epoch and MODELS_DIR/pytorch/best_model.pt when validation accuracy improves.
    """
    train_loader, val_loader, _ = loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Optimizer
    optimizer_name = cfg['optimizer']['name']
    lr = cfg['training']['learning_rate'] # Base config has this
    weight_decay = float(cfg['optimizer']['weight_decay'])

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Default to SGD if not adam, or raise error
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Loss
    loss_name = cfg['loss']['name']
    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss() # Default

    epochs = cfg['training']['epochs']

    # Logging setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(OUTPUTS_DIR, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info(f"Starting training on device: {device}")

    best_acc = 0.0
    models_dir = os.path.join(MODELS_DIR, "pytorch")
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        val_acc, val_loss = evaluate(model, val_loader, device)

        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pt"))
            logging.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Save last model
        torch.save(model.state_dict(), os.path.join(models_dir, "last_model.pt"))

    logging.info("Training complete.")

def main_pytorch():
    """
    Orchestrates configuration loading, data loader creation, model construction, and starts the PyTorch training run.
    
    Loads the project configuration (expected to merge base settings with the PyTorch-specific config), builds training/validation/test dataloaders, instantiates the model, and invokes the training routine with those objects.
    """
    from src.common.utils import load_config

    # Load config
    # We need to merge base.yaml and pytorch.yaml
    # Assuming load_config handles merging if we pass both or if we pass the specific one and it inherits.
    # The instruction says "Values must merge with base.yaml".
    # Let's assume load_config takes a list or we load base then update with pytorch.
    # But the instruction example shows `cfg = load_config("configs/pytorch.yaml")`.
    # This implies `load_config` might handle inheritance or we just load pytorch.yaml and it should have everything?
    # Or maybe we need to manually merge.
    # Let's check `src/common/utils.py` if it exists.
    # I'll assume for now `load_config` does the right thing or I'll implement a simple merge here if needed.
    # But I can't see `src/common/utils.py`.
    # I'll assume `load_config` is available as per instructions.

    # Wait, I haven't implemented `src/common/utils.py`.
    # The instructions for Phase 1 Task 1 say "Values must merge with base.yaml".
    # And Task 7 says "from src.pytorch.train import main_pytorch".
    # And the user rules say "Gemini MUST load configs via a clean helper: from src.common.utils import load_config".
    # I should probably check if `src/common/utils.py` exists.
    # `list_dir` of `src/common` showed 2 children.
    # Let's check `src/common` content.

    # I will write the function assuming `load_config` works as expected or I will fix it if I find it doesn't exist.
    # But for `train.py`, I need to import it.

    cfg = load_config("configs/pytorch.yaml")

    # Create dataloaders
    loaders = get_dataloaders(cfg)

    # Create model
    model = create_model(cfg)

    # Train
    train_model(cfg, model, loaders)

if __name__ == "__main__":
    main_pytorch()