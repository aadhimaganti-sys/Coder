from torch import nn


def linear_head(features: int, classes: int) -> nn.Linear:
    return nn.Linear(features, classes)
