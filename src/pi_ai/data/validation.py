from pathlib import Path
from PIL import Image


def find_corrupt_images(root: str | Path) -> list[Path]:
    corrupt = []
    for path in Path(root).rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            corrupt.append(path)
    return corrupt
