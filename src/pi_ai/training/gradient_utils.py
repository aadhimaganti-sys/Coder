from torch import nn


def clip_gradients(model: nn.Module, maximum_norm: float) -> None:
    nn.utils.clip_grad_norm_(model.parameters(), maximum_norm)
