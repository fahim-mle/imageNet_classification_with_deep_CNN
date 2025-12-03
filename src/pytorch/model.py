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
    num_classes = cfg['model'].get('num_classes', 10)

    if model_name == 'resnet18':
        # weights parameter is the new way to specify pretrained in newer torchvision
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet18(weights=weights)

        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'baseline':
        model = BaselineCNN(num_classes=num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    return model

class BaselineCNN(nn.Module):
    """
    A simple baseline CNN with 3 convolutional layers.
    """
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
