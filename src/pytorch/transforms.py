from torchvision import transforms

def get_transforms(image_size):
    """
    Build torchvision transform pipelines for training, validation, and testing.
    
    Parameters:
        image_size (int or tuple): Target image size as an int (square) or (height, width). If an int is provided it is converted to (image_size, image_size).
    
    Returns:
        tuple: (train_transforms, val_transforms, test_transforms) where each element is a torchvision.transforms.Compose pipeline. The training pipeline includes random horizontal flip; all pipelines perform resize, conversion to tensor, and normalization with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms, test_transforms