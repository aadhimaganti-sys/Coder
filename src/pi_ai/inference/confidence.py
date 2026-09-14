import torch


def confidence(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=1).amax(dim=1)
