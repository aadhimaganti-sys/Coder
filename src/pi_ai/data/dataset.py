from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from pi_ai.constants import SUPPORTED_IMAGE_SUFFIXES
from pi_ai.exceptions import DatasetError


class FolderImageDataset(Dataset[tuple[Tensor, int]]):
    """Lazy folder dataset using data/raw/class_name/image files."""

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetError(f"dataset directory does not exist: {self.root}")
        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.samples = [(path, self.class_to_index[path.parent.name]) for path in self.root.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
        if not self.samples:
            raise DatasetError(f"no supported images found below {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB" if self.transform is None else image.mode)
            tensor = self.transform(image) if self.transform else image
        return tensor, target
