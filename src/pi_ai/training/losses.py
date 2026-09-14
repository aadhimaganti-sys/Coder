from torch import nn


def cross_entropy() -> nn.Module:
    return nn.CrossEntropyLoss()
