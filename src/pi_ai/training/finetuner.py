from torch import nn


def freeze_features(model: nn.Module, freeze: bool = True) -> None:
    if not hasattr(model, "features"):
        raise ValueError("model has no feature extractor")
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
