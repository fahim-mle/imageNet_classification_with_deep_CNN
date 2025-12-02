from torchvision import transforms

def get_transforms(image_size):
    """
    Returns train, val, and test transforms.

    Args:
        image_size (tuple or int): Size of the image (height, width) or int for square.

    Returns:
        tuple: (train_transforms, val_transforms, test_transforms)
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
