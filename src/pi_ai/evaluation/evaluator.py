import torch
from torch import nn
from pi_ai.training.train_loop import run_epoch


def evaluate(model: nn.Module, loader, device: str = "cpu") -> dict[str, float]:
    return run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device(device))
