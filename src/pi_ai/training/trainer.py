from pathlib import Path
import torch
from torch import nn
from .checkpoint import save_checkpoint
from .losses import cross_entropy
from .optimizer import build_optimizer
from .train_loop import run_epoch


class Trainer:
    def __init__(self, model: nn.Module, device: str = "cpu", learning_rate: float = 1e-3) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = cross_entropy()
        self.optimizer = build_optimizer(self.model, learning_rate)

    def fit(self, train_loader, validation_loader=None, epochs: int = 1, checkpoint: str | Path | None = None) -> list[dict[str, float]]:
        history = []
        for epoch in range(epochs):
            metrics = run_epoch(self.model, train_loader, self.loss_fn, self.device, self.optimizer)
            if validation_loader is not None:
                metrics.update({f"val_{key}": value for key, value in run_epoch(self.model, validation_loader, self.loss_fn, self.device).items()})
            history.append(metrics)
            if checkpoint:
                save_checkpoint(checkpoint, self.model, self.optimizer, epoch, metrics)
        return history
