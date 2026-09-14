import torch
from pi_ai.training.train_loop import run_epoch


def test_eval_loop(model, batch):
    targets = torch.zeros(4, dtype=torch.long)
    result = run_epoch(model, [(batch, targets)], torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert 0 <= result["accuracy"] <= 1
