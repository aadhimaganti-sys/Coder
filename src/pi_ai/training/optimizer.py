from torch import nn, optim


def build_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate)
