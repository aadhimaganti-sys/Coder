from dataclasses import dataclass
import torch


@dataclass
class ClassificationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def classification_metrics(loss: float, predictions: torch.Tensor, targets: torch.Tensor, classes: int) -> ClassificationMetrics:
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[target.long(), prediction.long()] += 1
    true_positive = matrix.diag().float()
    precision = (true_positive / matrix.sum(0).clamp_min(1)).mean().item()
    recall = (true_positive / matrix.sum(1).clamp_min(1)).mean().item()
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return ClassificationMetrics(loss, (predictions == targets).float().mean().item(), precision, recall, f1)
