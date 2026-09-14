from torch.utils.data import DataLoader, Dataset


def build_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = False, workers: int = 0, pin_memory: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=pin_memory)
