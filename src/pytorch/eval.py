import torch

def evaluate(model, loader, device):
    """
    Evaluates the model on the given loader.

    Args:
        model: PyTorch model.
        loader: DataLoader.
        device: Device to run evaluation on.

    Returns:
        tuple: (accuracy, avg_loss)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / total

    return accuracy, avg_loss
