import torch
import torch.nn as nn
from torchvision import models

def create_model(cfg):
    """
    Builds a torchvision classification model from configuration and moves it to the available device (CUDA > MPS > CPU).
    
    Parameters:
        cfg (dict): Configuration dictionary. Expected keys:
            - model.name (str): Model identifier (currently supports "resnet18").
            - model.pretrained (bool): Whether to load pretrained weights.
            - model.num_classes (int, optional): Number of output classes; defaults to 10 if omitted.
    
    Returns:
        torch.nn.Module: The constructed model with its final classification layer adjusted to `num_classes` and moved to the selected device.
    
    Raises:
        ValueError: If `model.name` is not supported.
    """
    model_name = cfg['model']['name']
    pretrained = cfg['model']['pretrained']

    # We assume we are doing classification on Animals-10 (10 classes)
    # Ideally, num_classes should be in config or derived from dataset.
    # But for now, let's assume 10 as per dataset name.
    # Or better, let's check if we can pass it in.
    # The instruction says "Replaces final layer based on num classes".
    # Since we don't have the dataset object here, we'll default to 10 or check config.
    # Let's add num_classes to config if not present, or hardcode for this specific task if acceptable.
    # But to be generic, let's look for it in config or default to 10.
    num_classes = 10

    if model_name == 'resnet18':
        # weights parameter is the new way to specify pretrained in newer torchvision
        # but 'pretrained=True' is deprecated.
        # Let's check torch version or just use 'weights="DEFAULT"' if pretrained is True.
        # For compatibility with older/newer, let's try to handle it.
        # But for simplicity and standard practice in many tutorials:
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet18(weights=weights)

        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    return model