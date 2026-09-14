import torch
from pi_ai.training.train_loop import run_epoch


def test_training_step(model, batch):
    targets = torch.randint(3, (4,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, [(batch, targets)], torch.nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
    assert metrics["loss"] > 0
