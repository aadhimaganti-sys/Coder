import torch
from pi_ai.inference.batching import batches


def test_batches():
    assert len(list(batches(torch.zeros(5, 1), 2))) == 3
