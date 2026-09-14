import torch
from pi_ai.evaluation import evaluate


def test_evaluation(model, batch):
    result = evaluate(model, [(batch, torch.zeros(4, dtype=torch.long))])
    assert "loss" in result
