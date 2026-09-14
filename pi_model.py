"""Compact PyTorch image classifier suitable for Raspberry Pi 5.

The default demo uses 1 x 28 x 28 inputs and synthetic data so this file can
be executed immediately. For a real project, pass tensors with the same shape
to ``train_model`` and ``predict`` or replace ``make_demo_loaders``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class DepthwiseSeparableBlock(nn.Module):
    """A regular lightweight convolution block.

    The depthwise convolution processes one channel at a time, then a 1 x 1
    pointwise convolution mixes channels. This uses substantially fewer MACs
    than a full convolution while retaining local spatial information.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class TinyPiCNN(nn.Module):
    """Small CNN with bounded feature maps and global average pooling.

    The network has no large flattened hidden layer. Global average pooling
    makes the classifier independent of the exact spatial size and avoids a
    potentially expensive fully connected layer.
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def configure_cpu(threads: int | None = None) -> None:
    """Set conservative CPU threading to reduce contention and heat.

    A single or small number of threads is often more efficient on a Pi than
    launching many workers. Set ``threads`` to the measured best value.
    """

    thread_count = threads or min(2, os.cpu_count() or 1)
    torch.set_num_threads(max(1, thread_count))
    torch.set_num_interop_threads(1)


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.detach().item() * batch_size
            correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            examples += batch_size

    if examples == 0:
        raise ValueError("The data loader is empty")
    return EpochMetrics(total_loss / examples, correct / examples)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> list[EpochMetrics]:
    """Train with small batches and no retained computation graphs."""

    target_device = torch.device(device)
    model.to(target_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        history.append(_run_epoch(model, train_loader, loss_fn, target_device, optimizer))
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> EpochMetrics:
    """Evaluate without autograd tensors or gradient memory."""

    target_device = torch.device(device)
    model.to(target_device)
    return _run_epoch(model, data_loader, nn.CrossEntropyLoss(), target_device)


def predict(model: nn.Module, inputs: Tensor, device: str = "cpu") -> Tensor:
    """Return class indices using inference-only execution."""

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    with torch.inference_mode():
        return model(inputs.to(target_device)).argmax(dim=1).cpu()


def make_demo_loaders(
    samples: int = 256,
    batch_size: int = 32,
    num_classes: int = 10,
) -> DataLoader:
    """Create a deterministic, dependency-free loader for a smoke test."""

    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    targets = torch.randint(num_classes, (samples,), generator=generator)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test the compact Pi CNN")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    configure_cpu(args.threads)
    loader = make_demo_loaders(batch_size=args.batch_size)
    model = TinyPiCNN(num_classes=10)
    history = train_model(model, loader, epochs=args.epochs)
    metrics = evaluate_model(model, loader)
    sample_predictions = predict(model, torch.randn(4, 1, 28, 28))

    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"last train loss: {history[-1].loss:.4f}")
    print(f"demo accuracy: {metrics.accuracy:.2%}")
    print(f"sample predictions: {sample_predictions.tolist()}")


if __name__ == "__main__":
    main()