import os

import torch


def configure_cpu(threads: int = 2) -> None:
    """Use bounded threading to reduce CPU contention and heat."""
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def resolve_device(requested: str = "cpu") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cpu_count() -> int:
    return os.cpu_count() or 1
