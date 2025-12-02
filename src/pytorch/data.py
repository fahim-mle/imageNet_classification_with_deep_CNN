import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from src.pytorch.transforms import get_transforms
from src.common.paths import DATA_DIR

def get_dataloaders(cfg):
    """
    Creates and returns dataloaders for train, val, and test sets.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    image_size = cfg['dataset']['image_size']
    batch_size = cfg['training']['batch_size']
    processed_dir = os.path.join(DATA_DIR, "processed") # Assuming processed data is here, or raw if using ImageFolder directly on raw
    # The instructions say "Processed paths from config", but base.yaml has "processed_dir: data/processed"
    # Usually ImageFolder expects a directory structure class/image.jpg
    # If the data is already split in processed_dir (train/val/test), we should use that.
    # If not, we might need to split it.
    # Let's assume for now we are loading from a single directory and splitting, OR loading from split directories.
    # Looking at base.yaml: split_ratios are defined. This suggests we might need to split.
    # However, standard practice with ImageFolder is often to have train/val folders.
    # Let's check if we can find where the data is.
    # For now, I will implement a generic split approach if only one folder exists, or load from separate folders if they exist.
    # Actually, the instruction says "Processed paths from config".
    # Let's assume the data is in `processed_dir` and organized by class, and we need to split it.

    # Wait, if I look at base.yaml, it has split_ratios.
    # Let's assume we load the whole dataset and split it.

    data_path = os.path.join(DATA_DIR, "raw", "animals-10") # Or processed? base.yaml says raw_dir: data/raw, processed_dir: data/processed
    # The user might have downloaded data to raw.
    # Let's assume we use the 'raw' data for now if 'processed' is empty or we are doing on-the-fly loading.
    # But usually 'processed' implies it's ready.
    # Let's use the path from config.

    # Actually, let's look at the previous conversation summary. "Setting Up Data Pipeline".
    # It seems data might be in `data/raw`.
    # Let's check `data` directory content first to be sure.
    # But I need to write the file now.
    # I will write a robust version that tries to find the data.

    # Re-reading instructions: "Uses: datasets.ImageFolder, Processed paths from config".
    # Config has `processed_dir`.

    train_transforms, val_transforms, test_transforms = get_transforms(image_size)

    # We'll assume the dataset is in cfg['dataset']['raw_dir'] or 'processed_dir'.
    # Let's try to load from processed_dir first.
    dataset_dir = cfg['dataset']['processed_dir']
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
         dataset_dir = cfg['dataset']['raw_dir']

    # If the dataset is already split into train/val/test folders
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    else:
        # Load entire dataset and split
        full_dataset = datasets.ImageFolder(dataset_dir, transform=None) # Apply transforms after split?
        # It's hard to apply different transforms after split if we use random_split on a single dataset object.
        # Common workaround: Wrapper class or just use train_transforms for all if simple, or just subset.
        # A better way for this assignment if folders don't exist:
        # Create subsets and override __getitem__ or just use train_transforms for now if we have to.
        # But let's stick to the likely scenario: The user probably has data in a folder structure.
        # If not, we will split.

        # Let's assume for now we split.
        train_ratio = cfg['dataset']['split_ratios']['train']
        val_ratio = cfg['dataset']['split_ratios']['val']
        test_ratio = cfg['dataset']['split_ratios']['test']

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(cfg['dataset']['seed']))

        # We need to apply transforms.
        # Since we can't easily change transform of a subset's underlying dataset without affecting others if they share it,
        # we can create a simple wrapper or just accept that we might use same transform.
        # OR, we can re-instantiate ImageFolder 3 times (inefficient but works).

        # Let's use a helper class to apply transforms
        class TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.subset)

        train_dataset = TransformedSubset(train_subset, train_transforms)
        val_dataset = TransformedSubset(val_subset, val_transforms)
        test_dataset = TransformedSubset(test_subset, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
