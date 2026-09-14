from torch import nn
from .parameter_count import count_parameters, model_size_mb


def summarize(model: nn.Module) -> dict[str, float | int]:
    return {"parameters": count_parameters(model), "trainable": count_parameters(model, True), "size_mb": model_size_mb(model)}
