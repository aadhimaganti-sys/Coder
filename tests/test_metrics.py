import torch
from pi_ai.training.metrics import classification_metrics


def test_metrics():
    result = classification_metrics(1.0, torch.tensor([0, 1]), torch.tensor([0, 1]), 2)
    assert result.accuracy == 1
