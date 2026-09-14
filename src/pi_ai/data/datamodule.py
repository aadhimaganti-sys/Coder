from pathlib import Path
from torch.utils.data import DataLoader, random_split
from .dataset import FolderImageDataset
from .loaders import build_loader


def folder_loaders(root: str | Path, transform, batch_size: int = 32, workers: int = 0) -> tuple[DataLoader, DataLoader]:
    dataset = FolderImageDataset(root, transform)
    train_size = max(1, int(len(dataset) * 0.8))
    validation_size = len(dataset) - train_size
    if validation_size == 0:
        train_size -= 1
        validation_size = 1
    train, validation = random_split(dataset, [train_size, validation_size])
    return build_loader(train, batch_size, True, workers), build_loader(validation, batch_size, False, workers)
