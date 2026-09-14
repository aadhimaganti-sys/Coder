from typing import Callable

import torch
from PIL import Image


def image_transform(size: int, channels: int = 1) -> Callable:
    def transform(image: Image.Image) -> torch.Tensor:
        converted = image.convert("L" if channels == 1 else "RGB").resize((size, size))
        values = torch.frombuffer(bytearray(converted.tobytes()), dtype=torch.uint8).float().div_(255)
        return values.reshape(channels, size, size)
    return transform
