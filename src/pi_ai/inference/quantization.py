import torch
from torch import nn


def quantize_linear_layers(model: nn.Module) -> nn.Module:
    """Quantize supported Linear layers; convolution layers remain float32."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
