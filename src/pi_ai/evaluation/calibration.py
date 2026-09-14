import torch


def expected_confidence(logits: torch.Tensor) -> float:
    return float(logits.softmax(1).amax(1).mean())
