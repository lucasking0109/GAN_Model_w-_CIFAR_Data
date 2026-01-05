"""
Dataset Loading Module
Supports CIFAR-10 and CelebA datasets
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_dataloader(
    dataset_name: str = "cifar10",
    data_root: str = "./data_cache",
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    target_class: int = None,
):
    """
    Get data loader

    Args:
        dataset_name: Dataset name ("cifar10" or "celeba")
        data_root: Data storage path
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading worker threads
        target_class: Specific class to train (only for CIFAR-10)
                     CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
                                      5=dog, 6=frog, 7=horse, 8=ship, 9=truck
                     CelebA does not need this parameter

    Returns:
        DataLoader: Training data loader
    """
    # Define image transforms
    # CelebA needs CenterCrop then Resize (original is 178x218)
    # CIFAR-10 only needs Resize (original is 32x32)
    if dataset_name.lower() == "celeba":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])
    else:
        # CIFAR-10 and other datasets
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

    # CIFAR-10 class name mapping
    cifar10_classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name.lower() == "celeba":
        dataset = datasets.CelebA(
            root=data_root,
            split='train',
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Filter to specific class if specified (only for CIFAR-10)
    if target_class is not None and dataset_name.lower() == "cifar10":
        if target_class not in range(10):
            raise ValueError(f"Class must be between 0-9, got: {target_class}")

        # Find all indices belonging to target class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
        dataset = Subset(dataset, class_indices)

        print(f"Loading dataset: {dataset_name} (using only class {target_class}: {cifar10_classes[target_class]})")
    elif dataset_name.lower() == "celeba":
        print(f"Loading dataset: {dataset_name} (CelebA face dataset)")
    else:
        print(f"Loading dataset: {dataset_name} (using all classes)")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speed up GPU data transfer
        drop_last=True,   # Drop incomplete last batch
    )

    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {len(dataloader)}")
    print(f"  - Image size: {image_size}x{image_size}")

    return dataloader
