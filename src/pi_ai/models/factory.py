from torch import nn
from .mlp import TinyMLP
from .tiny_cnn import TinyPiCNN


def build_model(name: str, num_classes: int, input_channels: int = 1, input_features: int = 784) -> nn.Module:
    if name == "tiny_cnn":
        return TinyPiCNN(num_classes, input_channels)
    if name == "tiny_mlp":
        return TinyMLP(input_features, num_classes)
    raise ValueError(f"unknown model: {name}")
