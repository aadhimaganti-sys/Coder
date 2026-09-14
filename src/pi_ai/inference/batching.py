import torch


def batches(inputs: torch.Tensor, batch_size: int):
    for start in range(0, len(inputs), batch_size):
        yield inputs[start:start + batch_size]
