import torch
from torch.utils.data import TensorDataset
from pi_ai.data.loaders import build_loader


def test_loader_batch():
    loader = build_loader(TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)), 2)
    assert next(iter(loader))[0].shape[0] == 2
