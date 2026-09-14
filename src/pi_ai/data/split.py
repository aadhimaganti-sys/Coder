import random
from typing import Sequence


def split_indices(length: int, validation_fraction: float = 0.2, seed: int = 7) -> tuple[list[int], list[int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    split = max(1, int(length * (1 - validation_fraction)))
    return indices[:split], indices[split:]
