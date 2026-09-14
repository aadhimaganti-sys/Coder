from typing import Iterable
import torch
from torch import nn


def run_epoch(model: nn.Module, loader: Iterable, loss_fn: nn.Module, device: torch.device, optimizer=None, clip_norm: float | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    correct = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            count = targets.size(0)
            total_loss += loss.detach().item() * count
            total += count
            correct += (logits.detach().argmax(1) == targets).sum().item()
    if total == 0:
        raise ValueError("loader is empty")
    return {"loss": total_loss / total, "accuracy": correct / total}
