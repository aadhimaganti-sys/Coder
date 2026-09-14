from pathlib import Path
from PIL import Image


def open_image(path: str | Path, channels: int = 1) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L" if channels == 1 else "RGB").copy()
