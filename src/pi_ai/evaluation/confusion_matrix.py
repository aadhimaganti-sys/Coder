import torch


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, classes: int) -> torch.Tensor:
    result = torch.zeros(classes, classes, dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        result[target.long(), prediction.long()] += 1
    return result
