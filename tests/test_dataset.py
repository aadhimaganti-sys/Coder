from pathlib import Path
from PIL import Image
from pi_ai.data.dataset import FolderImageDataset
from pi_ai.data.transforms import image_transform


def test_folder_dataset(tmp_path: Path):
    path = tmp_path / "cat"
    path.mkdir()
    Image.new("L", (8, 8)).save(path / "one.png")
    dataset = FolderImageDataset(tmp_path, image_transform(8, 1))
    assert dataset[0][0].shape == (1, 8, 8)
